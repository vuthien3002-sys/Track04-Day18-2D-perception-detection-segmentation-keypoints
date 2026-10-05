# Báo cáo bonus — Lab Ngày 18

Link notebook đã chạy: https://github.com/vuthien3002-sys/Track04-Day18-2D-perception-detection-segmentation-keypoints/blob/main/lab_2d_perception_student.ipynb

Mọi số liệu dưới đây lấy từ output của `lab_2d_perception_student.ipynb` (Restart & Run All trên Colab, GPU Tesla T4,
`ultralytics==8.4.171`) và từ `submission/ket_qua.json`.

## 1D — `average_precision`

- Hàm đạt cả 4 phép thử của `check_average_precision()`: mọi dự đoán đều đúng → AP = 1, không có TP → AP = 0,
  kết quả không đổi khi xáo thứ tự đầu vào, và ví dụ trên slide (10 GT, 15 dự đoán).
- Đường PR của ví dụ trên slide cho **AP = 0.535**. Recall chỉ lên tới 0.7, nên 30 mức recall cuối trong 101 mức được tính 0.
- `ket_qua.json`: `progress.average_precision = "ok"`.

## 4C — Tập val lật gương: metric nào che lỗi `flip_idx`?

### Thiết lập

| | Model giải phẫu | Model đồng nhất |
|---|---|---|
| `flip_idx` | `[0, 1, 2, 3, 7, 6, 5, 4, 10, 11, 8, 9]` | `[0, 1, …, 11]` (YAML gốc) |
| Train | 40 epoch, imgsz 640, batch 16, seed 0, T4 | giống hệt |

**Val lật gương:** 53 ảnh val được lật ngang, x → 1 − x, rồi đổi chỗ keypoint theo `FLIP_IDX` giải phẫu. Tập này giả lập
hổ quay trái khi triển khai, điều mà tập val gốc không có (53 quay phải / 0 quay trái).

### Kết quả (53 ảnh)

| Model | Tập val | Box mAP50 | Box mAP50-95 | Pose mAP50 | Pose mAP50-95 |
|---|---|---:|---:|---:|---:|
| `flip_idx` giải phẫu | gốc | 0.995 | 0.930 | 0.995 | **0.457** |
| `flip_idx` giải phẫu | lật gương | 0.995 | 0.918 | 0.995 | **0.439** |
| `flip_idx` đồng nhất | gốc | 0.995 | 0.904 | 0.995 | **0.417** |
| `flip_idx` đồng nhất | lật gương | 0.995 | 0.894 | 0.878 | **0.298** |

### Metric nào đã che lỗi

- **Pose mAP50 trên val gốc** che lỗi hoàn toàn: cả hai model đều đạt 0.995. Mọi con hổ trong val đều quay phải, nên chân
  phía camera luôn là `right_*` và không có ca nào kiểm tra việc phân biệt trái/phải. Ngưỡng OKS 0.5 lại rộng nên metric
  bão hoà.
- **Box mAP** không bao giờ thấy lỗi này: Box mAP50 là 0.995 ở cả bốn dòng, vì box không mang thông tin trái/phải.
- **Pose mAP50-95 trên val gốc** chỉ hé lộ một chút (0.417 so với 0.457). Lúc train, ảnh bị lật ngang mà nhãn không đổi
  chỗ trái/phải, nên model đồng nhất học từ nhãn mâu thuẫn.
- **Val lật gương làm lỗi lộ rõ:** model đồng nhất tụt từ 0.417 xuống **0.298** Pose mAP50-95 (−29%), và Pose mAP50 cũng
  rơi xuống 0.878. Model giải phẫu gần như giữ nguyên: 0.457 → 0.439 (−4%).

### Bài học

Tập val phải có đủ các hướng mà model sẽ gặp khi triển khai. Ở đây nên bổ sung val lật gương với nhãn chuyển theo quy ước
giải phẫu, hoặc thu thêm ảnh hổ quay trái. Khi đánh giá pose, cần đọc Pose mAP50-95 hoặc OKS theo từng keypoint chứ
không chỉ Pose mAP50 hay Box mAP, vì hai metric này bão hoà và không nhạy với việc đảo trái/phải.
