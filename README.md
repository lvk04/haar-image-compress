Đồ án cho môn Phương pháp số cho Khoa học dữ liệu (MTH10607)
# Trình Nén Ảnh Bằng Biến Đổi Haar

Chương trình Python để **nén ảnh bằng biến đổi wavelet Haar**, hỗ trợ:
- Chọn ngưỡng nén thủ công (`threshold`)
- Hoặc tìm ngưỡng tự động theo **mức PSNR mong muốn**

---

## 🚀 Tính năng

- Nén ảnh bằng biến đổi Haar 2D.
- Hỗ trợ 2 chế độ nén:
  - **Nén cơ bản** với threshold cố định
  - **Nén nâng cao** tự động tìm threshold để đạt PSNR mục tiêu
- Hiển thị ảnh đầu vào và ảnh nén
- In ra thông tin nén: tỷ lệ nén, PSNR, kích thước ảnh
- Lưu ảnh nén vào thư mục tùy chọn

---
## 🖥️ Cách sử dụng

Chạy chương trình từ dòng lệnh:

```bash
python haar_matrix.py [đường_dẫn_ảnh] [-p PSNR_mục_tiêu] [-t threshold_cố_định] [-o thư_mục_lưu_ảnh]

