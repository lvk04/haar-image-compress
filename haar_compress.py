import numpy as np
from PIL import Image
import os
import argparse
import sys

class HaarCompressor:
    def __init__(self, output_dir=None):
        self.output_dir = output_dir
        self.threshold = None
        self.image_path = None
        self.levels = None
        self.target_psnr = None
        self.best_params = None
        self.original_image = None
        self.compressed_image = None
        self.compression_stats = None
        self.last_save_path = None

    def _ensure_numpy_image(self, image_input):
        """Chuyển ảnh, hoặc đường dẫn ảnh thành numpy array.
        """
        # Nếu đã là numpy array
        if isinstance(image_input, np.ndarray):
            return image_input
        # Nếu là PIL Image
        elif isinstance(image_input, Image.Image):
            img = image_input
        # Nếu là đường dẫn
        elif isinstance(image_input, str):
            try:
                img = Image.open(image_input)
            except Exception as e:
                raise ValueError(f"Cannot read image from: {image_input} ({e})")
        else:
            raise ValueError("This format is not supported")
        # Giữ nguyên nếu grayscale
        if img.mode == 'L':
            return np.array(img)
        # Chuyển các chế độ khác sang RGB
        return np.array(img.convert("RGB"))

    def _read_image(self):
        return self._ensure_numpy_image(self.image_path)

    @staticmethod
    def haar_1d(signal):
        """Hàm biến đổi Haar 1D."""
        factor = 1 / np.sqrt(2)
        avg = (signal[::2] + signal[1::2]) * factor
        diff = (signal[::2] - signal[1::2]) * factor
        return np.concatenate([avg, diff])

    @staticmethod
    def inverse_haar_1d(transformed):
        """Hàm biến đổi ngược Haar 1D."""
        n = len(transformed)
        factor = 1 / np.sqrt(2)
        avg = transformed[:n//2]
        diff = transformed[n//2:]
        recon = np.zeros(n)
        recon[::2] = (avg + diff) * factor
        recon[1::2] = (avg - diff) * factor
        return recon

    def haar_2d(self, channel):
        """Hàm biến đổi Haar 2D, bằng cách áp dụng Haar 1D theo hàng và cột."""
        h, w = channel.shape
        result = channel.copy().astype(float)
        for _ in range(self.levels):
            for i in range(h):
                result[i, :w] = self.haar_1d(result[i, :w])
            for j in range(w):
                result[:h, j] = self.haar_1d(result[:h, j])
            h //= 2
            w //= 2
        return result

    def inverse_haar_2d(self, transformed):
        """Hàm biến đổi ngược Haar 2D, bằng cách áp dụng biến đổi ngược Haar 1D theo hàng và cột."""
        H, W = transformed.shape
        result = transformed.copy().astype(float)
        curr_h = H // (2 ** self.levels)
        curr_w = W // (2 ** self.levels)
        for _ in range(self.levels):
            curr_h *= 2
            curr_w *= 2
            for j in range(curr_w):
                result[:curr_h, j] = self.inverse_haar_1d(result[:curr_h, j])
            for i in range(curr_h):
                result[i, :curr_w] = self.inverse_haar_1d(result[i, :curr_w])
        return result

    def pad_to_pow2(self, img):
        """Pad hình ảnh đến kích thước lũy thừa của 2.
        Trả về ảnh đã đệm, chiều cao và chiều rộng ban đầu."""
        h, w = img.shape
        new_h = 1 << int(np.ceil(np.log2(h)))
        new_w = 1 << int(np.ceil(np.log2(w)))
        pad_h = new_h - h
        pad_w = new_w - w
        self.levels = int(np.log2(min(new_h, new_w)))
        padded = np.pad(img, ((0, pad_h), (0, pad_w)), mode='constant')
        return padded, h, w

    def compress_channel(self, chan, threshold):
        """
        Nén một kênh ảnh và trả về các thông số đánh giá ảnh
        """
        padded, h, w = self.pad_to_pow2(chan)
        coeffs = self.haar_2d(padded)
        compressed = coeffs * (np.abs(coeffs) > threshold)
        recon_full = self.inverse_haar_2d(compressed)
        recon = recon_full[:h, :w]
        mse = np.mean((chan - recon) ** 2)
        psnr = (10 * np.log10(255**2 / mse)) if mse > 0 else float('inf')
        zero_pct = 100 * np.sum(compressed == 0) / compressed.size
        return compressed, recon, psnr, zero_pct

    def optimize_threshold(self, img_array):
        """
        Tối ưu ngưỡng nén dựa trên PSNR và tỷ lệ hệ số về 0:
            1.Tạo list threshold: Nếu threshold là số thì tạo list ừ 0 đến số đó, nếu None thì tạo từ 0 đến 200 với bước 5.
            2. Khởi tạo kết quả tốt nhất: Đặt best là ngưỡng chưa xác định, avg_zero = -1.
            3. Duyệt qua từng threshold t:
                - Nén từng kênh ảnh: Dùng compress_channel để tính PSNR và tỷ lệ hệ số bằng 0.
                - Tính các metrics avg_psnr và avg_zero từ các kênh.
                - Cập nhật kết quả tốt nhất: Nếu avg_psnr đạt yêu cầu và avg_zero lớn hơn trước đó.
            Trả kết quả: Trả về ngưỡng tối ưu cùng PSNR và % hệ số 0 tương ứng.
        """
        if isinstance(self.threshold, (int, float)):
            self.threshold = np.arange(0, self.threshold, 5)
        elif self.threshold is None:
            self.threshold = np.arange(0, 200, 5)

        best = {'threshold': None, 'avg_zero': -1, 'psnr': 0}
        chans = img_array if isinstance(img_array, list) else [img_array]

        for t in self.threshold:
            zeros, psnrs = [], []
            for chan in chans:
                _, _, psnr, zero_pct = self.compress_channel(chan, t)
                psnrs.append(psnr)
                zeros.append(zero_pct)

            avg_psnr = np.mean(psnrs)
            avg_zero = np.mean(zeros)

            if avg_psnr >= self.target_psnr and avg_zero > best['avg_zero']:
                best.update({'threshold': t, 'avg_zero': avg_zero, 'psnr': avg_psnr})

        return best

    def compress_image(self, image_path, target_psnr=30, threshold=None):
        """
        Nén ảnh bằng biến đổi Haar 2D dựa trên metric PSNR 
        args:
            image_path: Đường dẫn đến ảnh cần nén.
            target_psnr: Giá trị PSNR mục tiêu để tìm threshold tốt nhất.
            threshold: Ngưỡng cố định nếu không muốn tối ưu PSNR.
            
        1. Khởi tạo thông số: Lưu đường dẫn ảnh, PSNR mục tiêu, và threshold đầu vào.
        2. Đọc ảnh: Đọc ảnh và tách các kênh (1 kênh nếu ảnh xám, 3 nếu ảnh màu).
        3. Tối ưu threshold: Tìm threshold tốt nhất đạt PSNR yêu cầu bằng optimize_threshold.
        4. Kiểm tra threshold: Nếu không tìm được threshold phù hợp, báo lỗi.
        5. Nén từng kênh: Dùng threshold tốt nhất để nén và khôi phục từng kênh, lưu PSNR và % hệ số 0.
        6. Ghép lại các kênh thành ảnh hoàn chỉnh, chuyển về kiểu uint8.
        7. Lưu ảnh nén, các thông số và metrics.
        8. Trả về ảnh nén dưới dạng numpy array
        """
        self.image_path = image_path
        self.target_psnr = target_psnr
        self.threshold = threshold
        self.original_image = self._read_image()
        channels = [self.original_image] if self.original_image.ndim == 2 else [self.original_image[:, :, c] for c in range(3)]

        best = self.optimize_threshold(channels)
        if best['threshold'] is None:
            raise ValueError(f"No threshold found to achieve PSNR >= {self.target_psnr} dB.")

        recon_ch, psnr_list, zero_list = [], [], []
        for chan in channels:
            _, recon, psnr, zero_pct = self.compress_channel(chan, best['threshold'])
            recon_ch.append(recon)
            psnr_list.append(psnr)
            zero_list.append(zero_pct)

        recon_img = recon_ch[0] if len(recon_ch) == 1 else np.stack(recon_ch, axis=2)
        self.compressed_image = np.clip(recon_img, 0, 255).astype(np.uint8)
        self.best_params = best
        self.compression_stats = {'psnr_list': psnr_list, 'zero_list': zero_list}
        return self.compressed_image

    def compress_image_by_threshold(self, image_input, threshold):
        """
        Hàm nén ảnh bằng biến đổi Haar với threshold cố định.
        Thuật toán:
        1. Đọc ảnh từ đường dẫn hoặc numpy array.
        2. Chia ảnh thành các kênh (nếu là ảnh màu).
        3. Nén từng kênh bằng biến đổi Haar với ngưỡng đã cho.
        4. Tính PSNR và tỷ lệ phần trăm hệ số bằng 0 cho từng kênh.
        5. Tính trung bình PSNR và tỷ lệ phần trăm hệ số bằng 0.
        6. Trả về ảnh nén và các thông số thống kê.
        """
        if isinstance(image_input, str):
            self.image_path = image_input
        else:
            self.image_path = None
        self.original_image = self._ensure_numpy_image(image_input)
        channels = [self.original_image] if self.original_image.ndim == 2 else [self.original_image[:, :, c] for c in range(3)]
        recon_ch, psnr_list, zero_list = [], [], []
        for chan in channels:
            _, recon, psnr, zero_pct = self.compress_channel(chan, threshold)
            recon_ch.append(recon)
            psnr_list.append(psnr)
            zero_list.append(zero_pct)
        # Compute averages and print
        avg_psnr = float(np.mean(psnr_list))
        avg_zero = float(np.mean(zero_list))
        print(f"Average PSNR: {avg_psnr:.2f} dB")
        print(f"Average Zero Percentage: {avg_zero:.2f}%")
        compressed_image = recon_ch[0] if len(recon_ch) == 1 else np.stack(recon_ch, axis=2)
        self.compressed_image = np.clip(compressed_image, 0, 255).astype(np.uint8)
        self.best_params = {'threshold': threshold}
        self.compression_stats = {'psnr_list': psnr_list,'psnr': avg_psnr, 'zero_list': zero_list, 'average_zero': avg_zero}
        return self.compressed_image

    def display_results(self):
        """Hiển thị ảnh gốc và ảnh đã nén."""
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        titles = ['Original', 'Reconstructed']
        images = [self.original_image, self.compressed_image]

        for ax, img, title in zip(axs, images, titles):
            ax.imshow(img.astype(np.uint8))
            ax.set_title(title)
            ax.axis('off')

        plt.tight_layout()
        plt.show()

    def save_reconstructed_image(self):
        """Lưu ảnh đã nén vào thư mục đầu ra hoặc thư mục hiện tại nếu không có."""
        name, ext = os.path.splitext(os.path.basename(self.image_path))
        out_dir = self.output_dir or os.getcwd()
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{name}_recon{ext}")

        mode = 'L' if self.compressed_image.ndim == 2 else 'RGB'
        Image.fromarray(self.compressed_image, mode=mode).save(out_path)
        self.last_save_path = out_path
        return out_path


    def get_compression_info(self):
        if not self.best_params:
            return None
        return {
            'levels': self.levels,
            'target_psnr': self.target_psnr,
            **self.best_params,
            **self.compression_stats
        }


def main():
    """Chương trình chính để nén ảnh bằng biến đổi Haar."""
    parser = argparse.ArgumentParser(
        description="Compress images using Haar wavelet transform with optional PSNR target or fixed threshold."
    )
    parser.add_argument('image_path', help='Path to the input image')
    parser.add_argument('-p', '--psnr', type=float,
                        help='Target PSNR value for automatic threshold selection')
    parser.add_argument('-t', '--threshold', type=float,
                        help='Fixed threshold for compression (overrides PSNR target)')
    parser.add_argument('-o', '--output_dir', type=str,
                        help='Directory to save the reconstructed image')
    args = parser.parse_args()

    compressor = HaarCompressor(output_dir=args.output_dir)
    try:
        if args.psnr is not None:
            if args.threshold is not None:
                print("Both PSNR and threshold provided. Compression based on PSNR.")
                compressor.compress_image(args.image_path, target_psnr=args.psnr)
            else:
                compressor.compress_image(args.image_path, target_psnr = args.psnr)
        if args.psnr is None:
            if args.threshold is None: 
                compressor.compress_image_by_threshold(args.image_path, threshold=30)  # Default threshold if none provided
            else:
                compressor.compress_image_by_threshold(args.image_path, threshold=args.threshold)  

        compressor.display_results()
        out_path = compressor.save_reconstructed_image()
        info = compressor.get_compression_info()
        print(f"Reconstructed image saved to: {out_path}")
        print("Compression stats:")
        for k, v in info.items():
            print(f"  {k}: {v}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()

# Cách chạy chương trình:
#Chạy câu lệnh sau trong terminal:
#python haar_matrix.py [đường_dẫn_ảnh_đầu_vào] [-p] [Mức PSNR mục tiêu] [-t] [Mức Threshold tối đa] [-o] [Đường_dẫn_đầu_ra]

#Ví dụ:
#python haar_matrix.py hcmus.jpg -p 30 -t 50 
#- Nếu trong câu lệnh có PSNR mục tiêu, ứng dụng sẽ nén theo cách nâng cao - tìm threshold tối ưu để đạt được mức PSNR đó. 
#Nếu có,  \texttt{Mức Threshold} sẽ là threshold tối đa ứng dụng sẽ thử. Nếu không có sẽ sử dụng threshold mặc định.
#-Nếu trong câu lệnh không có PSNR mục tiêu, Ứng dụng sẽ nén theo cách cơ bản: dựa theo threshold đã cung cấp.
#  Nếu không có cả \texttt{Threshold} thì sẽ nén dựa trên threshold mặc định.