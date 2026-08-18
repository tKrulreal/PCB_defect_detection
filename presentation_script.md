# 🎓 KỊCH BẢN BẢO VỆ ĐỒ ÁN TỐT NGHIỆP CHI TIẾT (MASTER DEFENSE SCRIPT)

- **Đề tài:** Phát hiện và phân loại lỗi bo mạch in (PCB) sử dụng mô hình Hybrid YOLOv8 kết hợp CNN và cơ chế giải thích Grad-CAM (XAI)
- **Sinh viên thực hiện:** Nguyễn Duy Khương
- **Mã số sinh viên (MSSV):** 11236134
- **Đơn vị đào tạo:** Trường Đại học Công nghệ — Đại học Kinh tế Quốc dân (NEU)
- **Chuyên ngành:** Công nghệ Thông tin
- **Tổng thời lượng bảo vệ:** 12 – 15 phút thuyết trình & demo + 10 – 15 phút phản biện Q&A

---

## ⏱️ PHÂN BỔ THỜI LƯỢNG THUYẾT TRÌNH (TỔNG: 13 PHÚT)

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│ 00:00 - 01:30 (1.5 phút) │ SLIDE 1 - 2  : Mở đầu, Giới thiệu & Điểm nghẽn sản xuất SMT           │
│ 01:30 - 04:30 (3.0 phút) │ SLIDE 3 - 6  : Kiến trúc Hybrid 2 giai đoạn (YOLOv8 + CNN Refinement) │
│ 04:30 - 06:30 (2.0 phút) │ SLIDE 7 - 9  : XAI Grad-CAM, Đánh giá Hungarian & Pipeline Code       │
│ 06:30 - 10:00 (3.5 phút) │ SLIDE 10 ➔ 🌐: LIVE DEMO THỰC CHIẾN TRÊN STREAMLIT DASHBOARD          │
│ 10:00 - 12:00 (2.0 phút) │ SLIDE 11 - 12: Phân tích lỗi Root Cause, Giải pháp & Lộ trình Edge AI │
│ 12:00 - 12:30 (0.5 phút) │ SLIDE 13     : Tổng kết, Lời cảm ơn & Mời Hội đồng đặt câu hỏi       │
│ 12:30+                   : PHIÊN VẤN ĐÁP PHẢN BIỆN VỚI HỘI ĐỒNG (Q&A SESSION)                   │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🎙️ KỊCH BẢN CHI TIẾT TỪNG SLIDE (WORD-BY-WORD SCRIPT)

---

### 📍 SLIDE 1: TRANG TIÊU ĐỀ & TỔNG QUAN ĐỀ TÀI (00:00 – 00:45)
- **Hành động:** Đứng thẳng, phong thái tự tin, mắt bao quát toàn bộ Hội đồng, mỉm cười nhẹ và cúi đầu chào.
- **Lời nói:**
  > *"Kính thưa quý Thầy, Cô trong Hội đồng đánh giá khóa luận tốt nghiệp,*
  >
  > *Kính thưa Thầy/Cô hướng dẫn cùng toàn thể các bạn sinh viên có mặt trong buổi bảo vệ ngày hôm nay.*
  >
  > *Em tên là **Nguyễn Duy Khương**, sinh viên chuyên ngành Công nghệ Thông tin, Trường Đại học Công nghệ — Đại học Kinh tế Quốc dân.*
  >
  > *Hôm nay, em xin phép được báo cáo đề tài khóa luận tốt nghiệp của mình với tiêu đề: **'Hệ thống phát hiện và phân loại lỗi bo mạch in (PCB) sử dụng mô hình Hybrid YOLOv8 kết hợp CNN và cơ chế giải thích Grad-CAM'**.*
  >
  > *Đề tài tập trung giải quyết bài toán kiểm định chất lượng quang học tự động trong công nghiệp sản xuất điện tử với 4 chỉ số cốt lõi: Độ chính xác định vị mAP@50 đạt **99.46%**, độ chính xác phân loại chi tiết đạt **98.79% đến 100%**, thuật toán ghép cặp Hungarian đạt **75.67%**, và bao phủ toàn diện **6/6 dạng khuyết tật mạch** kết hợp cơ chế giải thích trực quan Grad-CAM.*
  >
  > *Sau đây, em xin phép được bắt đầu phần trình bày chi tiết."*
- **Thao tác:** Nhấn `Phím Space` hoặc `Mũi tên phải` để chuyển sang Slide 2.

---

### 📍 SLIDE 2: BỐI CẢNH & THÁCH THỨC CÔNG NGHIỆP (00:45 – 01:30)
- **Hành động:** Chỉ tay vào bảng so sánh 3 phương pháp bên phải màn hình.
- **Lời nói:**
  > *"Kính thưa Hội đồng, trong các dây chuyền dán bề mặt SMT hiện đại, kiểm tra bo mạch in là khâu sống còn quyết định độ tin cậy của thiết bị điện tử. Tuy nhiên, các nhà máy hiện nay đang đối mặt với 3 điểm nghẽn lớn:*
  >
  > *1. **Kiểm tra thủ công (MVI):** Tốn 15 đến 30 giây cho mỗi bo mạch. Sau 2 giờ làm việc liên tục, công nhân bị mỏi mắt dẫn đến tỷ lệ bỏ sót lỗi chủ quan từ **20% đến 30%**.*
  >
  > *2. **Máy AOI truyền thống dựa trên luật (Rule-based AOI):** Sử dụng phương pháp trừ ảnh mẫu nhị phân. Phương pháp này cực kỳ nhạy cảm với ánh sáng và độ rung, dẫn tới **tỷ lệ báo động giả (False Positive) lên tới 40% - 60%**, làm quá tải trạm kiểm duyệt lại của nhà máy.*
  >
  > *3. **Thách thức về tỷ lệ kích thước (Scale Imbalance):** Các khuyết tật như hở mạch, gặm mép hay gai đồng thường có kích thước siêu nhỏ, chiếm **chưa tới 0.1% diện tích bo mạch**.*
  >
  > *Nhìn vào bảng so sánh bên phải, giải pháp AI Hybrid mà em đề xuất giải quyết trọn vẹn cả 3 bài toán: Đạt tốc độ thời gian thực **~33 ms/bo mạch**, độ chính xác trên crop đạt **98.79%**, tỷ lệ báo giả dưới **2.1%**, và có bản đồ nhiệt Grad-CAM giúp kỹ sư kiểm chứng minh bạch."*
- **Thao tác:** Nhấn `Phím Space` chuyển sang Slide 3.

---

### 📍 SLIDE 3: TỔNG QUAN GIẢI PHÁP & KIẾN TRÚC HYBRID (01:30 – 02:30)
- **Hành động:** Rê laser pointer theo luồng 4 bước từ trái sang phải trên sơ đồ khối.
- **Lời nói:**
  > *"Để khắc phục triệt để các hạn chế trên, em đề xuất kiến trúc **Hybrid 2 giai đoạn** theo nguyên lý 'Chia để trị' (Divide and Conquer):*
  >
  > - ***Bước 1 (Input):** Tiếp nhận ảnh bo mạch quang học độ phân giải cao.*
  > - ***Bước 2 (Stage 1 - Localization):** Sử dụng mạng YOLOv8m để quét nhanh toàn bộ bức ảnh và định vị tọa độ các vùng có nguy cơ khuyết tật, đạt mAP@50 là **99.46%**.*
  > - ***Bước 3 (Dynamic Cropping & Padding):** Tự động cắt từng bounding box khả nghi, đồng thời mở rộng biên **25% margin padding** (tối thiểu 32x32 pixel) để bảo toàn trọn vẹn ngữ cảnh đường dây đồng xung quanh.*
  > - ***Bước 4 (Stage 2 - Fine-Grained Classification & XAI):** Đưa từng crop ảnh vào mạng Deep CNN chuyên biệt (ResNet-18) để tái phân loại chính xác 6 lớp lỗi, đồng thời trích xuất bản đồ nhiệt Grad-CAM.*
  >
  > *Đặc biệt, ở góc phải màn hình là **Công thức tích hợp độ tin cậy kép**:*
  > $$C_{\text{final}} = C_{\text{YOLO}} \times C_{\text{CNN}}$$
  > *Việc nhân xác suất giữa bộ dò vị trí và bộ phân loại tạo nên một 'bộ lọc kép' thông minh, tự động triệt tiêu các đề xuất nhiễu từ Stage 1."*
- **Thao tác:** Nhấn `Phím Space` chuyển sang Slide 4.

---

### 📍 SLIDE 4: BỘ DỮ LIỆU & 6 LOẠI LỖI PCB ĐẶC TRƯNG (02:30 – 03:15)
- **Hành động:** Chỉ vào lưới 6 thẻ màu sắc đặc trưng của 6 loại lỗi.
- **Lời nói:**
  > *"Nghiên cứu được thực hiện trên bộ dữ liệu chuẩn **PKU PCB Defect Dataset** với 1,386 ảnh bo mạch gốc và **2,158 bounding box ground truth** bao phủ 6 loại lỗi cơ bản theo tiêu chuẩn quốc tế IPC-A-610:*
  >
  > 1. ***missing_hole (Màu đỏ):** Lỗi thiếu lỗ khoan via hoặc chân linh kiện THT do gãy mũi khoan CNC.*
  > 2. ***mouse_bite (Màu cam):** Lỗi gặm mép đường dây đồng dạng răng cưa do ăn mòn hóa học quá mức.*
  > 3. ***open_circuit (Màu vàng):** Lỗi đứt hở mạch hoàn toàn, làm mất kết nối tín hiệu điện.*
  > 4. ***short (Màu xanh lá):** Lỗi đoản mạch, cầu nối đồng thừa nối chập 2 đường dây liền kề.*
  > 5. ***spur (Màu xanh dương):** Lỗi gai đồng nhọn nhô ra từ cạnh đường mạch.*
  > 6. ***spurious_copper (Màu tím):** Lỗi đốm đồng thừa cô lập trên nền phíp cách điện FR4.*
  >
  > *Dữ liệu được làm giàu (Data Augmentation) gấp 4 lần thông qua xoay góc $\pm 10^{\circ}$, biến đổi độ sáng/tương phản ColorJitter $\pm 20\%$ và lật gương để đảm bảo mô hình có tính bất biến cao với điều kiện ánh sáng nhà xưởng."*
- **Thao tác:** Nhấn `Phím Space` chuyển sang Slide 5.

---

### 📍 SLIDE 5: GIAI ĐOẠN 1 — YOLOV8 DEFECT LOCALIZATION (03:15 – 04:00)
- **Hành động:** Chỉ vào 4 thẻ chỉ số lớn ở cột bên phải.
- **Lời nói:**
  > *"Ở Giai đoạn 1, em lựa chọn kiến trúc **YOLOv8m** với các cải tiến kỹ thuật quan trọng:*
  >
  > - *Sử dụng Backbone **CSPDarknet kết hợp khối C2f** giúp tối ưu dòng truyền gradient đa nhánh.*
  > - *Cấu trúc **PANet + FPN** đa tỷ lệ kết hợp 3 tầng đặc trưng P3, P4, P5 giúp bắt trọn các lỗi từ kích thước siêu nhỏ vài pixel đến các lỗi lớn.*
  > - *Đầu ra **Anchor-Free Decoupled Head** kết hợp hàm mất mát hồi quy tọa độ CIoU và DFL Loss.*
  >
  > *Mô hình được huấn luyện ở kích thước ảnh **768×768 px**, sử dụng Optimizer **AdamW** với Learning Rate $8 \times 10^{-4}$ kết hợp Cosine Annealing trong 150 epochs.*
  >
  > *Kết quả thực nghiệm trên tập Test đạt thành tích xuất sắc: **mAP@50 đạt 99.46%**, Precision đạt **99.23%**, và Recall đạt **99.08%**. Tốc độ suy luận chỉ mất **~18.2 ms/ảnh**, đảm bảo không bỏ sót bất kỳ ứng viên lỗi nào trước khi chuyển giao sang Stage 2."*
- **Thao tác:** Nhấn `Phím Space` chuyển sang Slide 6.

---

### 📍 SLIDE 6: GIAI ĐOẠN 2 — CNN REFINEMENT & SO SÁNH KIẾN TRÚC (04:00 – 04:45)
- **Hành động:** Hướng mắt vào bảng so sánh 3 mô hình Deep CNN.
- **Lời nói:**
  > *"Sau khi Stage 1 định vị bounding box, Stage 2 sử dụng cơ chế **Dynamic Crop mở rộng biên 25%** để đưa vào phân loại chi tiết.*
  >
  > *Em đã tiến hành thực nghiệm so sánh độc lập giữa 3 kiến trúc mạng nơ-ron sâu phổ biến nhất hiện nay:*
  >
  > 1. ***ResNet-18:** Đạt độ chính xác tuyệt đối **100.0%** trên tập test, dung lượng nhẹ **11.18 triệu tham số**, và tốc độ suy luận siêu tốc **2.89 ms/ảnh**.*
  > 2. ***ResNet-50:** Đạt Test Accuracy **99.91%**, tham số 23.52M, độ trễ 6.58 ms.*
  > 3. ***EfficientNet-B2:** Đạt Test Accuracy **99.81%**, tham số 7.71M, độ trễ 14.55 ms.*
  >
  > *Quá trình huấn luyện áp dụng kỹ thuật **Differential Learning Rate** (học tốc độ chậm $3 \times 10^{-4}$ ở Backbone và $1 \times 10^{-3}$ ở Classifier Head) kết hợp **OneCycleLR** và **Label Smoothing ($\epsilon=0.05$)**.*
  >
  > *Dựa trên kết quả thực nghiệm, **ResNet-18** được lựa chọn làm mô hình mặc định nhờ sự cân bằng hoàn hảo giữa độ chính xác tuyệt đối 100% và tốc độ suy luận nhanh gấp 5 lần so với EfficientNet-B2."*
- **Thao tác:** Nhấn `Phím Space` chuyển sang Slide 7.

---

### 📍 SLIDE 7: EXPLAINABLE AI — GRAD-CAM TRỰC QUAN HÓA (04:45 – 05:30)
- **Hành động:** Chỉ vào 3 khung trực quan hóa (Panel 1 ➔ Panel 2 ➔ Panel 3).
- **Lời nói:**
  > *"Một trong những đóng góp mang tính thực tiễn cao nhất của đồ án là **xóa bỏ tính chất 'Hộp đen' (Black-box)** của mạng nơ-ron sâu trong môi trường sản xuất công nghiệp nhờ cơ chế Explainable AI với **Grad-CAM**.*
  >
  > *Về mặt toán học: Thuật toán tính đạo hàm riêng của điểm số lớp dự đoán $Y^c$ đối với feature map $A^k$ tại tầng tích chập cuối cùng (`model.layer4[-1]`), từ đó xác định trọng số tầm quan trọng $\alpha_k^c$ và tổng hợp qua hàm ReLU:*
  > $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_k \alpha_k^c A^k \right)$$
  >
  > *Nhìn vào 3 khung trực quan hóa trên màn hình:*
  > - *Khung 1 là ảnh vết cắt vi mô gốc.*
  > - *Khung 2 là bản đồ nhiệt kích hoạt Jet colormap.*
  > - *Khung 3 là ảnh phủ lớp mờ Blended Overlay ($\alpha = 0.45$).*
  >
  > *Thầy Cô có thể thấy vùng đỏ rực hội tụ chính xác $100\%$ vào đúng mép đồng bị gặm của lỗi `mouse_bite`, không bị phân tán ra nền phíp FR4 hay các lỗ via. Điều này mang lại căn cứ khoa học vững chắc để kỹ sư QA/QC phê duyệt kết quả kiểm định."*
- **Thao tác:** Nhấn `Phím Space` chuyển sang Slide 8.

---

### 📍 SLIDE 8: ĐÁNH GIÁ TOÀN DIỆN HỆ THỐNG & HUNGARIAN MATCHING (05:30 – 06:15)
- **Hành động:** Chỉ vào biểu đồ phân bố thanh F1-score của 6 lớp lỗi.
- **Lời nói:**
  > *"Để đánh giá khách quan toàn bộ hệ thống End-to-End mà không bị sai số trùng lặp, em áp dụng **Thuật toán ghép cặp đồ thị 2 phía Hungarian (Kuhn-Munkres)** với điều kiện ràng buộc IoU $\ge 0.50$ giữa box dự đoán và ground truth:*
  >
  > - *Trên toàn bộ **2,158 bounding box ground truth** của tập test: Hệ thống đề xuất 1,689 box và khớp chính xác **1,633 True Positive boxes**.*
  > - *Độ chính xác ghép cặp toàn hệ thống đạt **75.67%**, trong khi độ chính xác phân loại trên các vùng crop tìm được đạt **98.79%**.*
  > - *Điểm **Macro F1-Score đạt 0.9880**, trong đó lớp `missing_hole` dẫn đầu với F1 **0.992**, lớp `short` đạt **0.989**, và các lớp còn lại đều duy trì trên **0.975**.*
  >
  > *Về mặt tốc độ phần cứng: Khi chạy suy luận trên card đồ họa **NVIDIA GeForce RTX 4060 GPU**, toàn bộ pipeline chỉ mất **33.05 ms/ảnh (~30.2 FPS)**, đáp ứng trọn vẹn tiêu chuẩn tốc độ thời gian thực của băng chuyền SMT công nghiệp."*
- **Thao tác:** Nhấn `Phím Space` chuyển sang Slide 9.

---

### 📍 SLIDE 9: MÃ NGUỒN CỐT LÕI & KIỂM THỬ PHẦN MỀM (06:15 – 06:45)
- **Hành động:** Chỉ vào khung Terminal code `predict_image()` bên trái và thẻ Test suite bên phải.
- **Lời nói:**
  > *"Toàn bộ mã nguồn của hệ thống được đóng gói theo kiến trúc module hóa hướng đối tượng chuẩn mực trong file `stage12_yolo_cnn_system.py` với lớp `Stage12Pipeline`.*
  >
  > *Hàm `predict_image` thực hiện tuần tự: Đọc ảnh vào bộ nhớ RAM ➔ Chạy YOLO Stage 1 ➔ Dynamic Crop & Forward qua ResNet-18 ➔ Tính Combined Confidence và trả về cấu trúc JSON chuẩn.*
  >
  > *Đặc biệt, dự án đã xây dựng bộ kiểm thử tự động toàn diện với **58 Automated Unit & Integration Tests đạt tỷ lệ Pass 100%**, đảm bảo mã nguồn xử lý an toàn mọi trường hợp ngoại lệ như ảnh rỗng, tọa độ tràn biên ảnh, hoặc ma trận Hungarian rỗng."*
- **Thao tác:** Nhấn `Phím Space` chuyển sang Slide 10.

---

### 📍 SLIDE 10: ỨNG DỤNG WEB STREAMLIT & SMART FACTORY (06:45 – 07:15)
- **Hành động:** Dừng lại ở Slide 10, chỉ vào 4 khối chức năng và chuẩn bị thao tác chuyển sang trình duyệt web.
- **Lời nói:**
  > *"Thưa Hội đồng, đồ án không dừng lại ở mức mô hình nghiên cứu lý thuyết mà đã được đóng gói thành một **Ứng dụng Web Dashboard Streamlit hoàn chỉnh** sẵn sàng phục vụ nhà máy thông minh với 5 phân hệ:*
  >
  > 1. ***Tab 1 - Detector:*** Soi chi tiết đơn ảnh, Kính lúp vi mô $4\times - 8\times$, Grad-CAM XAI và xuất Phiếu QA/QC xuất xưởng.
  > 2. ***Tab 2 - SMT Live Stream:*** Mô phỏng camera quét tự động luồng 1,069 bo mạch thực tế trên GPU RTX 4060, tích hợp tháp đèn Andon Light và bộ đếm KPI sản lượng thời gian thực.
  > 3. ***Tab 3 - Batch Gallery:*** Kiểm định mẻ hàng loạt và thống kê lỗi.
  > 4. ***Tab 4 - Performance:*** 8 biểu đồ phân tích Confusion Matrix & Training Curves.
  > 5. ***Tab 5 - System Config:*** Tùy biến tham số phần cứng và mô hình.
  >
  > *Sau đây, em xin phép được chuyển sang giao diện trình duyệt để thực hiện phần **DEMO TRỰC TIẾP TRÊN HỆ THỐNG**."*

---

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│ 🔥 CHUYỂN CẢNH: NHẤN TỔ HỢP PHÍM [ALT + TAB] SANG TRÌNH DUYỆT WEB STREAMLIT (HTTP://LOCALHOST:8501)│
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🌐 KỊCH BẢN THAO TÁC LIVE DEMO CHI TIẾT TRÊN WEB (07:15 – 10:00, 3.5 PHÚT)

### 📍 BƯỚC DEMO 1: HEADER MONITOR, ĐỒNG HỒ HỆ THỐNG & GPU ACCELERATION (~30 GIÂY)
- **Hành động thao tác chuột:**
  1. Rê chuột chỉ vào thanh **Header Top Bar** có chấm xanh phát sáng và đồng hồ đang chạy từng giây.
  2. Mở rộng thanh **Sidebar bên trái**, chỉ vào dòng chữ: `🚀 Hardware: cuda (GPU - NVIDIA GeForce RTX 4060 Laptop GPU)`.
- **Lời nói:**
  > *"Kính thưa Thầy Cô, đây là giao diện trung tâm điều hành Smart Factory của hệ thống.*
  >
  > *Tại Header trên cùng, hệ thống tích hợp sẵn **Đồng hồ thời gian thực (Live System Clock)** và **Bộ đếm thời gian hoạt động liên tục (System Uptime)**.*
  >
  > *Trên thanh Sidebar bên trái, hệ thống tự động nhận diện và kích hoạt card đồ họa rời **NVIDIA GeForce RTX 4060 Laptop GPU** với CUDA cores chuyên dụng. Tại đây, kỹ sư có thể chuyển đổi nhanh giữa 3 mô hình ResNet-18, ResNet-50 hoặc EfficientNet-B2, đồng thời tinh chỉnh ngưỡng Confidence và NMS IoU theo điều kiện ánh sáng nhà xưởng."*

---

### 📍 BƯỚC DEMO 2: TAB 1 — SOI KÍNH LÚP VI MÔ 4X-8X & XUẤT PHIẾU QA/QC (~1.5 PHÚT)
- **Hành động thao tác chuột:**
  1. Tại Tab **`Detector (Soi Chi Tiết)`**, click vào nút ảnh mẫu nhanh: **`▸ l_light_01_missi`** (hoặc chọn 1 ảnh bất kỳ từ tab `📦 1,069 Bo Mạch Tập Test Dataset`).
  2. Bấm nút màu xanh: **`🚀 Bắt Đầu Kiểm Tra PCB Này`**.
  3. Chỉ vào thanh đo độ trễ: `⚡ Thời Gian Xử Lý AI: 34.2 ms (~29.2 FPS)`.
  4. Chọn radio button: **`🔍 Kính Lúp Soi Vi Mô Tương Tác (Micro-Zoom Loupe 4x-8x)`**.
  5. Kéo thanh trượt phóng đại lên **`6x`** hoặc **`8x`**, dùng chuột rê qua lại trên vết đứt mạch `open_circuit` và vết `mouse_bite` trên 2 khung ảnh.
  6. Cuộn xuống, mở rộng mục **`👁️ Chi Tiết Vùng Cắt & Bản Đồ Nhiệt Grad-CAM`**, chỉ vào bản đồ nhiệt đỏ rực.
  7. Cuộn xuống phần **`Phiếu Nghiệm Thu Chất Lượng QA/QC`**:
     - Bấm mở rộng **`👁️ Xem trước Phiếu Kiểm Định QA/QC Xuất Xưởng`**.
     - Chỉ vào con dấu mộc điện tử màu đỏ `[ ❌ REJECTED - DEFECTS FOUND ]` (hoặc màu xanh nếu bo PASS), tên kỹ sư kiểm định **Nguyễn Duy Khương - 11236134**, bảng kê tọa độ bounding box và nút in PDF.
- **Lời nói:**
  > *"Như Thầy Cô quan sát, khi em nhấn chạy kiểm tra, trên GPU RTX 4060 hệ thống chỉ mất đúng **34 mili-giây**, lập tức phát hiện chính xác vị trí khuyết tật.*
  >
  > *Nhằm hỗ trợ kỹ sư kiểm tra các chi tiết mạch siêu nhỏ dưới 10 pixel, em đã phát triển tính năng **Kính lúp vi mô tương tác (Interactive Micro-Zoom Loupe)**. Khi em rê chuột và kéo thanh phóng đại lên $6\times - 8\times$, tròng kính lúp tròn phát sáng neon sẽ phóng to đồng bộ cả ảnh quang học gốc và ảnh AI đã nhận diện, giúp soi rõ từng sợi dây đồng bị gặm mép hay đứt gãy.*
  >
  > *Phía dưới, bản đồ nhiệt **Grad-CAM XAI** hội tụ chính xác $100\%$ vào khuyết tật.*
  >
  > *Đặc biệt, hệ thống tự động sinh **Phiếu Chứng Nhận Chất Lượng QA/QC xuất xưởng** chuẩn quốc tế ISO/IPC-A-610 với đầy đủ mã lô sản xuất, UUID bo mạch, chữ ký kỹ sư kiểm định, dấu mộc điện tử và hỗ trợ xuất bản in PDF chỉ với 1 cú click chuột."*

---

### 📍 BƯỚC DEMO 3: TAB 2 — 🏭 SMT LIVE STREAM & THÁP ĐÈN ANDON LIGHT (~1.0 PHÚT)
- **Hành động thao tác chuột:**
  1. Click chuyển sang Tab 2: **`🏭 SMT Live Stream`**.
  2. Chỉ vào menu nguồn ảnh: `📦 Tập Test Thực Tế (pcb-defect-dataset/test/images - 1,069 bo mạch)`.
  3. Bật công tắc: **`▶️ Chạy Băng Chuyền`**.
  4. Thao tác: Quan sát các bo mạch chạy liên tục qua camera quang học:
     - Khi gặp bo sạch lỗi: Tháp đèn **Andon Tower** báo dải màu xanh lá `[ PASS - CONFORMANT ]`.
     - Khi gặp bo lỗi: Tháp đèn **Andon Alert** lập tức đổi sang viền đỏ nhấp nháy `[ REJECT / NG - DEFECTS DETECTED ]` kèm tên loại lỗi phát hiện.
     - Các thẻ KPI: Tổng số bo đã quét, Tỉ lệ PASS %, Tỉ lệ Lỗi %, và Tốc độ FPS nhảy số liên tục.
     - Cuộn xuống chỉ vào bảng **`Activity Feed Log`** ghi lại lịch sử 20 bo mạch vừa quét kèm dấu thời gian từng giây.
- **Lời nói:**
  > *"Tiếp theo, em xin phép trình diễn phân hệ **Mô phỏng Băng chuyền SMT tự động (SMT Live Stream)**.*
  >
  > *Hệ thống đang nạp trực tiếp toàn bộ **1,069 bo mạch thực tế** từ tập Test của dataset. Khi em bật công tắc vận hành băng chuyền, camera công nghiệp quét liên tục các bo mạch với tốc độ thời gian thực trên GPU RTX 4060.*
  >
  > *Khi bo mạch đạt chuẩn, **Tháp Đèn Andon Light** báo Xanh `PASS`. Khi phát hiện bo mạch có khuyết tật, tháp đèn lập tức chớp Đỏ cảnh báo `REJECT / NG`, đồng thời ghi nhận mã bo mạch và độ trễ vào bảng nhật ký sản lượng phía dưới. Tính năng này mô phỏng trọn vẹn quy trình kiểm định tự động trong các nhà máy thông minh Industry 4.0 hiện đại."*

---

### 📍 BƯỚC DEMO 4: TAB 3 & 4 — GALLERY HÀNG LOẠT & PERFORMANCE (~30 GIÂY)
- **Hành động thao tác chuột:**
  1. Click chuyển nhanh qua **Tab 3: Gallery** ➔ Lướt qua tính năng kiểm định hàng loạt.
  2. Click chuyển qua **Tab 4: Performance** ➔ Rê chuột qua ma trận nhầm lẫn Confusion Matrix và các biểu đồ Loss/F1.
  3. Nhấn tổ hợp phím `Alt + Tab` để quay trở lại Slide thuyết trình tại **Slide 11**.
- **Lời nói:**
  > *"Ngoài ra, Tab Gallery hỗ trợ nạp mẻ hàng trăm bo mạch cùng lúc để xuất báo cáo ca sản xuất, và Tab Performance tích hợp sẵn 8 biểu đồ theo dõi sức khỏe mô hình.*
  >
  > *Sau đây, em xin phép quay trở lại slide để đi vào phần phân tích nguyên nhân lỗi và lộ trình phát triển trong tương lai."*

---

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│ 🔥 CHUYỂN CẢNH: NHẤN TỔ HỢP PHÍM [ALT + TAB] QUAY TRỞ LẠI SLIDE THUYẾT TRÌNH TẠI SLIDE 11        │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### 📍 SLIDE 11: PHÂN TÍCH LỖI & GIẢI PHÁP KỸ THUẬT CẢI TIẾN (10:00 – 11:00)
- **Hành động:** Chỉ vào 3 khối phân tích lỗi bên trái và 3 giải pháp công nghiệp bên phải.
- **Lời nói:**
  > *"Kính thưa Hội đồng, từ file phân tích lỗi chi tiết `pipeline_errors.csv`, em đã phân loại các trường hợp lỗi thành 3 nhóm gốc rễ:*
  >
  > 1. ***Bỏ sót ở Stage 1 (False Negative):** Chiếm **87.2%** (505/579 lỗi). Nguyên nhân do các vết lỗi vi mô quá nhỏ dưới 10 pixel bị suy giảm tín hiệu sau các tầng pooling của YOLOv8.*
  > 2. ***Phân loại sai ở Stage 2 (Misclassification):** Chỉ chiếm **6.6%** (20-38 boxes), tập trung chủ yếu ở cặp nhầm lẫn hình thái tương đồng giữa `Spur` (Gai đồng nối mạch) và `Spurious Copper` (Đốm đồng cô lập sát mép).*
  > 3. ***Báo giả Bounding Box (False Positive):** Chiếm **6.2%** (36 boxes) do bóng đổ quang học từ các via pad khoan.*
  >
  > *Để khắc phục triệt để, em đề xuất 3 giải pháp nâng cấp công nghiệp ở cột bên phải:*
  > - *Tích hợp kỹ thuật **SAHI (Slicing Aided Hyper Inference)** cắt ảnh 2048px thành các ô trượt 640px giữ nguyên 100% độ phân giải ban đầu, giúp tăng khả năng bắt lỗi vi mô lên trên 95%.*
  > - *Áp dụng **Focal Loss & Hard Negative Mining** tăng trọng số cho các cặp lỗi khó.*
  > - *Sử dụng hệ thống **Chiếu sáng vòm đồng trục (Coaxial Dome Lighting)** để triệt tiêu hoàn toàn bóng đổ quang học."*
- **Thao tác:** Nhấn `Phím Space` chuyển sang Slide 12.

---

### 📍 SLIDE 12: TỔNG KẾT ĐỀ TÀI & LỘ TRÌNH PHÁT TRIỂN (11:00 – 12:00)
- **Hành động:** Chỉ vào 3 thành tựu cốt lõi và 4 giai đoạn lộ trình công nghiệp hóa.
- **Lời nói:**
  > *"Tổng kết lại, đồ án đã hoàn thành xuất sắc 3 mục tiêu cốt lõi:*
  > 1. *Xây dựng thành công **Hệ thống AI Hybrid 2 giai đoạn** đạt mAP50 99.46%, độ chính xác phân loại 98.79% - 100%, tốc độ thời gian thực ~33ms trên GPU.*
  > 2. *Tích hợp thành công **Explainable AI với Grad-CAM** giải thích minh bạch quyết định của mô hình.*
  > 3. *Đóng gói hoàn chỉnh ứng dụng **Web Dashboard Smart Factory** có mô phỏng băng chuyền SMT, tháp đèn Andon và xuất chứng chỉ QA/QC chuẩn ISO/IPC.*
  >
  > *Về lộ trình phát triển trong tương lai:*
  > - ***Q3/2026:** Lượng tử hóa **TensorRT FP16/INT8**, nhúng trực tiếp lên máy tính biên **NVIDIA Jetson Orin Nano** đạt độ trễ < 10ms.*
  > - ***Q4/2026:** Xây dựng vòng lặp **Active Learning** tự động thu thập các mẫu khó trong ca sản xuất để tái huấn luyện.*
  > - ***Q1/2027:** Kết nối giao thức công nghiệp **Modbus TCP / OPC-UA** điều khiển cánh tay robot SMT tự động gắp tách bo mạch lỗi.*
  > - ***Q2/2027:** Mở rộng sang hệ thống **3D Stereo AOI** đo độ cong vênh và chân hàn IC."*
- **Thao tác:** Nhấn `Phím Space` chuyển sang Slide 13.

---

### 📍 SLIDE 13: LỜI CẢM ƠN & PHIÊN VẤN ĐÁP Q&A (12:00 – 12:30)
- **Hành động:** Đứng trang nghiêm, mắt nhìn thẳng Hội đồng, mỉm cười và cúi đầu cảm ơn chân thành.
- **Lời nói:**
  > *"Kính thưa quý Thầy, Cô trong Hội đồng đánh giá và quý Thầy, Cô hướng dẫn,*
  >
  > *Trên đây là toàn bộ nội dung báo cáo và demo sản phẩm đồ án tốt nghiệp của em. Em xin trân trọng gửi lời cảm ơn sâu sắc nhất tới quý Thầy, Cô Khoa Công nghệ Thông tin — Trường Đại học Công nghệ, Đại học Kinh tế Quốc dân đã tận tình giảng dạy và định hướng cho em trong suốt quá trình học tập và thực hiện đề tài.*
  >
  > *Em xin kính mời quý Thầy, Cô trong Hội đồng cho ý kiến nhận xét và đặt câu hỏi phản biện để em được làm rõ hơn các khía cạnh kỹ thuật của đề tài.*
  >
  > *Em xin trân trọng cảm ơn!"*

---

## 🎯 BỘ 10 CÂU HỎI PHẢN BIỆN CHUYÊN SÂU & ĐÁP ÁN TRẢ LỜI CHUẨN CHUYÊN GIA

---

### ❓ CÂU HỎI 1: Tại sao không dùng YOLOv8 End-to-End một giai đoạn mà phải thiết kế kiến trúc Hybrid 2 giai đoạn phức tạp?
- **Cách trả lời tự tin:**
  > *"Em xin cảm ơn câu hỏi rất hay của Thầy/Cô ạ.*
  >
  > *Lý do cốt lõi bắt nguồn từ **hiện tượng mất mát đặc trưng không gian (Spatial Feature Loss)** của các mô hình Object Detection một giai đoạn:*
  > 1. *Trong ảnh bo mạch PCB độ phân giải cao (2048×2048), vết lỗi chỉ chiếm kích thước từ 8 đến 20 pixel (dưới 0.1% diện tích). Khi YOLOv8 downsample feature maps qua các tầng stride 8, 16, 32, tín hiệu của các vết lỗi siêu nhỏ này gần như bị hòa tan vào nền phíp cách điện.*
  > 2. *Nếu chỉ dùng YOLO đơn lẻ, mô hình phải vừa gánh nhiệm vụ hồi quy tọa độ vừa phải phân loại 6 lớp có hình thái cực kỳ giống nhau (như `Spur` và `Spurious Copper`), dẫn tới tỷ lệ nhầm lẫn cao.*
  > 3. *Bằng cách tách 2 giai đoạn: **Stage 1 YOLOv8 chỉ tập trung tối ưu Recall** để khoanh vùng candidate. Sau đó, **Dynamic Crop cắt box ra và phóng to lên kích thước chuẩn 224×224 px**, đưa vào mạng Deep CNN chuyên biệt. Lúc này, độ phân giải hiệu dụng của vết lỗi tăng lên gấp hàng chục lần, giúp mạng ResNet-18 trích xuất chi tiết vi mô và đạt độ chính xác phân loại lên tới **98.79% - 100%**."*

---

### ❓ CÂU HỎI 2: Tại sao độ chính xác phân loại trên Crop đạt 98.79% - 100%, nhưng độ chính xác ghép cặp toàn hệ thống (Hungarian System Accuracy) chỉ đạt 75.67%?
- **Cách trả lời tự tin:**
  > *"Em xin cảm ơn câu hỏi đào sâu về số liệu thực nghiệm của Thầy/Cô ạ.*
  >
  > *Sự chênh lệch giữa 2 con số này phản ánh đúng bản chất đánh giá khắt khe trong thị giác máy tính:*
  > 1. ***Crop Accuracy (98.79%):** Là độ chính xác của Stage 2 khi đã có sẵn ảnh crop chứa lỗi, đo lường năng lực phân loại thuần túy của mạng ResNet-18.*
  > 2. ***Hungarian System Accuracy (75.67%):** Là chỉ số đánh giá **toàn bộ Pipeline nối tiếp End-to-End**. Để được tính là 1 True Positive, hệ thống phải thỏa mãn đồng thời 3 điều kiện cực kỳ nghiêm ngặt:*
  >    - *Stage 1 phải dò trúng bounding box.*
  >    - *Độ trùng khớp không gian **IoU giữa box dự đoán và ground truth phải $\ge 0.50$**.*
  >    - *Stage 2 phải phân loại đúng nhãn.*
  > 3. *Qua phân tích lỗi `pipeline_errors.csv`, nguyên nhân khiến System Accuracy ở mức 75.67% là do **87.2% lỗi thuộc về việc Stage 1 bỏ sót các box siêu nhỏ (dưới 10px)**. Một khi Stage 1 đã bắt được box, Stage 2 phân loại chính xác gần như tuyệt đối (98.79%).*
  > *Đây chính là lý do em đề xuất giải pháp tích hợp SAHI (Slicing Aided Hyper Inference) trong lộ trình nâng cấp để đưa System Accuracy lên trên 90%."*

---

### ❓ CÂU HỎI 3: Cơ chế hoạt động của Grad-CAM là gì? Tại sao lại hook vào tầng tích chập cuối cùng (`layer4[-1]`) mà không phải tầng đầu?
- **Cách trả lời tự tin:**
  > *"Dạ em xin phép được giải thích nguyên lý hoạt động của Grad-CAM ạ:*
  >
  > 1. *Grad-CAM sử dụng **gradient của điểm số dự đoán lớp $c$ đối với các feature maps $A^k$** của tầng tích chập để đo lường mức độ ảnh hưởng của từng feature map tới quyết định phân loại.*
  > 2. *Trọng số tầm quan trọng $\alpha_k^c$ được tính bằng cách lấy trung bình toàn cục (Global Average Pooling) của các gradient. Sau đó, kết hợp tuyến tính các feature maps với trọng số $\alpha_k^c$ và đưa qua hàm kích hoạt **ReLU** để chỉ giữ lại các đặc trưng có đóng góp tích cực cho lớp $c$.*
  > 3. *Lý do hook vào **tầng tích chập cuối cùng (`model.layer4[-1]` ở ResNet)** là vì:*
  >    - *Các tầng đầu chỉ học các đặc trưng sơ cấp cấp thấp như đường nét, góc cạnh đơn giản.*
  >    - *Tầng tích chập cuối cùng là nơi lưu giữ **thông tin ngữ nghĩa cấp cao (High-level semantic features)** giàu ý nghĩa nhất, đồng thời vẫn giữ được thông tin không gian 2D (Spatial coordinates) của vật thể, từ đó tạo ra bản đồ nhiệt trực quan chuẩn xác nhất."*

---

### ❓ CÂU HỎI 4: Tại sao trong slide ghi tốc độ 30 FPS trên GPU RTX 4060, nhưng khi demo trên web đôi khi thấy độ trễ khoảng vài trăm mili-giây?
- **Cách trả lời tự tin:**
  > *"Em xin cảm ơn Thầy/Cô đã quan sát rất kỹ phần đo đạc độ trễ ạ.*
  >
  > *Có sự khác biệt rõ ràng giữa **Thời gian suy luận phần cứng AI thuần túy (Inference Latency)** và **Thời gian phản hồi toàn phần trên giao diện Web (End-to-End UI Overhead)**:*
  > 1. ***Tốc độ AI thuần túy trên GPU RTX 4060:** Khi chạy Forward Pass của YOLOv8m và ResNet-18 với CUDA, mô hình chỉ mất đúng **33.05 ms/ảnh (~30.2 FPS)**.*
  > 2. ***Độ trễ tính toán Grad-CAM:** Thuật toán Grad-CAM bắt buộc phải chạy một lượt **Backward Pass** để tính đạo hàm ngược qua mạng. Quá trình này tiêu tốn tài nguyên gấp 3-4 lần so với việc chỉ chạy suy luận Forward.*
  > 3. ***Độ trễ truyền tải Web Streamlit:** Streamlit chạy trên nền Python cục bộ, khi hiển thị ảnh lên trình duyệt, hệ thống phải mã hóa ảnh sang định dạng PNG/JPEG và truyền qua giao thức WebSocket.*
  > *Trong nhà máy thực tế, hệ thống sẽ chạy ở chế độ **Fast Inspection (~30 FPS)** để phân loại tự động; tính năng Grad-CAM và xuất phiếu PDF chỉ được kích hoạt khi kỹ sư cần soi chi tiết ở chế độ **Deep Review Station**."*

---

### ❓ CÂU HỎI 5: Tại sao trong 3 mô hình CNN lại chọn ResNet-18 thay vì ResNet-50 hay EfficientNet-B2 vốn có kiến trúc phức tạp hơn?
- **Cách trả lời tự tin:**
  > *"Dạ, em xin phép trả lời ạ:*
  >
  > *Trong kỹ thuật triển khai công nghiệp, nguyên tắc quan trọng nhất là **sự đánh đổi tối ưu giữa Độ chính xác (Accuracy) và Độ trễ (Latency)**:*
  > 1. *Tập dữ liệu crop lỗi PCB sau khi chuẩn hóa 224×224 px có đặc trưng hình thái tương đối rõ ràng. **ResNet-18 với 11.18 triệu tham số** đã đủ dung lượng biểu diễn để đạt độ chính xác **100.0% trên tập test độc lập** mà không bị hiện tượng Overfitting.*
  > 2. *ResNet-50 có 23.52M tham số (nặng gấp đôi) nhưng Test Accuracy chỉ đạt 99.91% và độ trễ tăng lên 6.58 ms.*
  > 3. *EfficientNet-B2 sử dụng các phép tích chập Depthwise Separable Convolutions, mặc dù số tham số ít (7.71M) nhưng trên phần cứng GPU lại bị nghẽn bộ nhớ do cấu trúc phân nhánh phức tạp, dẫn tới độ trễ lên tới 14.55 ms (chậm gấp 5 lần ResNet-18).*
  > *Do đó, **ResNet-18 là lựa chọn số 1 tuyệt đối** cho các hệ thống nhúng biên và dây chuyền thời gian thực."*

---

### ❓ CÂU HỎI 6: Công thức tính độ tin cậy kết hợp $C_{\text{final}} = C_{\text{YOLO}} \times C_{\text{CNN}}$ có ưu điểm gì so với việc lấy trung bình cộng?
- **Cách trả lời tự tin:**
  > *"Em xin cảm ơn câu hỏi rất thú vị của Thầy/Cô ạ.*
  >
  > *Lý do em chọn **phép nhân xác suất (Joint Probability Multiplication)** thay vì trung bình cộng là vì tính chất loại trừ nghiêm ngặt:*
  > 1. *Giả sử Stage 1 phát hiện một vùng nhiễu với $C_{\text{YOLO}} = 0.30$, nhưng Stage 2 nhận định đây là lỗi với $C_{\text{CNN}} = 0.90$.*
  >    - *Nếu lấy trung bình cộng: $\frac{0.30 + 0.90}{2} = 0.60$ (Vẫn vượt qua ngưỡng 0.5 và gây ra Báo giả - False Positive).*
  >    - *Nếu lấy tích xác suất: $0.30 \times 0.90 = 0.27$ (Lập tức bị loại bỏ dưới ngưỡng).*
  > 2. *Phép nhân đóng vai trò như một **cổng logic 'VÀ' (Logical AND Gate)**: Chỉ khi cả bộ định vị vị trí và bộ phân loại chi tiết đều có độ tin cậy cao thì kết quả mới được công nhận, giúp giảm thiểu tối đa tỷ lệ báo giả cho nhà máy."*

---

### ❓ CÂU HỎI 7: Cặp lỗi `Spur` (Gai đồng) và `Spurious Copper` (Đồng thừa) dễ bị nhầm lẫn nhất, nguyên nhân và cách khắc phục là gì?
- **Cách trả lời tự tin:**
  > *"Dạ thưa Thầy/Cô:*
  >
  > 1. ***Nguyên nhân hình thái:** `Spur` là mấu đồng dính liền vào đường dẫn, còn `Spurious Copper` là đốm đồng cô lập tự do. Khi đốm đồng thừa nằm cách đường dẫn chỉ 1 đến 2 pixel, sau quá trình downsample và padding, ranh giới phân tách bị mờ đi khiến mô hình dễ nhầm lẫn giữa gai dính liền và đốm rời.*
  > 2. ***Giải pháp khắc phục:** Em đề xuất 2 hướng:*
  >    - *Về thuật toán: Sử dụng hàm mất mát **Focal Loss** kết hợp **Hard Example Mining** để ép mô hình tập trung học các đặc trưng ranh giới khe hở (boundary gap).*
  >    - *Về xử lý ảnh: Tăng độ phân giải crop từ 224px lên 384px đối với riêng 2 lớp lỗi này để bảo toàn độ sắc nét của khe hở cách điện."*

---

### ❓ CÂU HỎI 8: Nếu điều kiện ánh sáng nhà xưởng thay đổi hoặc bo mạch bị đặt xoay góc thì hệ thống có hoạt động ổn định không?
- **Cách trả lời tự tin:**
  > *"Dạ thưa Thầy/Cô, hệ thống được thiết kế để đảm bảo tính bất biến cao với môi trường thực tế nhờ 3 yếu tố:*
  > 1. ***Data Augmentation đa dạng:** Trong quá trình huấn luyện, mô hình đã được nạp các mẫu xoay góc ngẫu nhiên $\pm 10^{\circ}$, biến đổi độ sáng và tương phản ColorJitter $\pm 20\%$.*
  > 2. ***Kiểm chứng trên tập xoay 270 độ:** Trong tập test của dataset có sẵn các mẫu `rotation_270` và mô hình vẫn nhận diện chính xác.*
  > 3. ***Giải pháp phần cứng:** Khi triển khai công nghiệp, hệ thống sẽ sử dụng nguồn sáng vòm đồng trục (Coaxial Dome Lighting) cung cấp cường độ sáng cố định và triệt tiêu bóng đổ, không phụ thuộc vào ánh sáng môi trường bên ngoài."*

---

### ❓ CÂU HỎI 9: Hệ thống kết nối với cánh tay robot gắp sản phẩm và phần mềm điều hành nhà máy MES/ERP như thế nào?
- **Cách trả lời tự tin:**
  > *"Dạ thưa Thầy Cô, hệ thống đã được thiết kế sẵn sàng cho việc tích hợp công nghiệp:*
  > 1. ***Kết nối tầng điều khiển (OT Layer):** Hệ thống giao tiếp với PLC (Siemens S7-1200 / Mitsubishi) thông qua giao thức **Modbus TCP** hoặc **OPC-UA**. Khi camera phát hiện bo lỗi (Andon Alert NG), hệ thống gửi tín hiệu bit sang thanh ghi PLC để kích hoạt xi-lanh khí nén hoặc cánh tay robot gắp tách bo lỗi ra khay NG.*
  > 2. ***Kết nối tầng quản trị (IT Layer):** Hàm `predict_image` tự động xuất dữ liệu lỗi chuẩn **JSON và XML**, có thể đẩy qua **REST API** hoặc **MQTT Broker** để lưu trữ trực tiếp vào hệ thống quản lý sản xuất MES / ERP theo thời gian thực."*

---

### ❓ CÂU HỎI 10: Ước tính chi phí phần cứng và bài toán hoàn vốn (ROI) khi triển khai giải pháp này vào một nhà máy SMT thực tế là bao nhiêu?
- **Cách trả lời tự tin:**
  > *"Dạ thưa Thầy Cô, đây là bài toán kinh tế rất khả thi:*
  > 1. ***Chi phí phần cứng:** Một trạm kiểm tra AI hoàn chỉnh gồm: 1 Camera công nghiệp Basler 5MP (~800 USD) + 1 Đèn vòm LED (~200 USD) + 1 Máy tính biên NVIDIA Jetson Orin Nano (~500 USD). Tổng chi phí phần cứng chỉ khoảng **~1,500 – 2,000 USD (khoảng 40 – 50 triệu VNĐ)**, rẻ hơn rất nhiều so với máy AOI ngoại nhập (thường có giá từ 30,000 đến 80,000 USD).*
  > 2. ***Bài toán hoàn vốn (ROI):** Một trạm kiểm tra tự động thay thế được 2 đến 3 công nhân kiểm tra thủ công cho mỗi ca sản xuất (tiết kiệm khoảng 20-30 triệu VNĐ/tháng). Như vậy, nhà máy chỉ mất **khoảng 2 đến 3 tháng là đã có thể thu hồi toàn bộ chi phí đầu tư ban đầu**, đồng thời giảm tỷ lệ sản phẩm lỗi lọt ra thị trường về mức gần như bằng 0."*

---

## 📋 BẢNG TỔNG HỢP CÁC PHÍM TẮT & THAO TÁC QUAN TRỌNG KHI BẢO VỆ

| Phím / Thao tác | Chức năng | Thời điểm sử dụng |
| :--- | :--- | :--- |
| **`F`** | Bật / Tắt chế độ Toàn màn hình (Fullscreen) | Nhấn ngay khi bắt đầu đứng lên bục thuyết trình |
| **`Space`** hoặc **`→`** | Chuyển tới slide kế tiếp | Thao tác chuyển slide theo nhịp nói |
| **`←`** | Quay lại slide trước | Khi Thầy Cô yêu cầu xem lại slide |
| **`Alt + Tab`** | Chuyển nhanh giữa Slide và Trình duyệt Web | Khi bắt đầu và kết thúc phần Live Demo |
| **`O`** | Bật chế độ xem tổng quan lưới 13 slide | Khi Thầy Cô hỏi bất kỳ slide nào trong phiên Q&A |
| **`B`** hoặc **`.`** | Tạm thời làm đen màn hình (Blackout) | Khi muốn toàn bộ Hội đồng tập trung vào lời nói |
