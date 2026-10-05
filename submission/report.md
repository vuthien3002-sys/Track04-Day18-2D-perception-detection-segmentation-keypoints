# Báo cáo nộp bài — Lab Ngày 18: 2D Perception

Link notebook đã chạy: https://github.com/vuthien3002-sys/Track04-Day18-2D-perception-detection-segmentation-keypoints/blob/main/lab_2d_perception_student.ipynb

Notebook nặng khoảng 12 MB vì còn nguyên output. Nếu GitHub không hiển thị được, mở bản xem qua nbviewer:
https://nbviewer.org/github/vuthien3002-sys/Track04-Day18-2D-perception-detection-segmentation-keypoints/blob/main/lab_2d_perception_student.ipynb

## Tóm tắt

- Chạy `Restart session and run all` trên Colab (GPU Tesla T4, `ultralytics==8.4.171`): 53 ô code chạy theo thứ tự,
  không ô nào lỗi.
- `final_report()`: mọi dòng bắt buộc đều ✅, `Câu hỏi: 12/12`. Không mượn phao: mọi mục trong `ket_qua.json` đều `"ok"`.
- 4B: fine-tune YOLO26n-pose 40 epoch, `imgsz=640` trên GPU. Pose mAP50 0.995, Pose mAP50-95 0.457.
- Bonus: 1D (AP = 0.535), 4C (val lật gương, hai model) và bài tập về nhà 3 (ONNX trên CPU).

## Các file trong `submission/`

| File | Nội dung |
|---|---|
| `ket_qua.json` | Kết quả kiểm tra do `final_report()` xuất |
| `autolabel/bus.txt` | Nhãn YOLO-seg của 5 object ở 2C, do `final_report()` chép sang |
| `bao_cao_bonus.md` | Bonus 1D và 4C |
| `bao_cao_bai_tap_ve_nha.md` | Báo cáo bài tập về nhà 3: export ONNX, đo latency trên CPU |
| `bai_tap_ve_nha_onnx.ipynb` | Notebook bài tập về nhà đã chạy trên Colab, còn nguyên output |

## Ghi chú

Bảng latency ở 1C thay đổi sau mỗi lần chạy. Sau lần chạy nộp bài, câu trả lời Q2 được cập nhật để dẫn đúng bảng trong
notebook này. Trong `ket_qua.json`, chỉ trường `answers.Q2` được sửa cho khớp với ô Q2 của notebook; mọi trường trạng
thái (`progress`) và kết quả tính toán vẫn giữ nguyên như `final_report()` đã xuất.
