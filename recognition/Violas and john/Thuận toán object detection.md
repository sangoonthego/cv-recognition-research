# Các thuận toán object detection 
**1. object detection:**
- là một nhánh của computer vision, mục tiêu là xác định vị trí và loại đối tượng trong một bức ảnh hoặc video.
- Một vài ví dụ về object detection : 
![ảnh minh họa](https://kajabi-storefronts-production.kajabi-cdn.com/kajabi-storefronts-production/file-uploads/blogs/22606/images/3fa2e-b5a1-06ca-5fe6-dd660dffd8a_crosswalk-1.webp "Tooltip khi hover")  
- Các nhánh của object detection : 
    1. Face Detection
    2. Face recognition
    3. Instance recognition
    4. Category recognition
## 2. Face Detection  
+ phát hiện vị trí khuôn mặt trong ảnh hoặc video (thường bằng bounding box). Đây là nhánh con nhận diện khuôn mặt con người có trong 1 bức ảnh hay không 
+ ảnh minh họa : 
![ảnh](https://developers.google.com/static/ml-kit/vision/face-detection/images/face_contours.svg)
### Thuật toán Truyền thống (Classical methods)  
1. **Viola–Jones (2001)**  
   - Thuật toán nổi tiếng nhất, mở đầu cho Face Detection real-time.  
   - Dùng:  
     - Haar-like features  
     - Integral Image  
     - AdaBoost  
     - Cascade Classifier  
   -  Ưu điểm: nhanh, chạy được trên máy yếu.  
   - Nhược điểm: kém chính xác trong điều kiện ánh sáng, góc nhìn phức tạp.  
   - Không phải deep learning.  

2. **HOG + SVM (2005)**  
   - Dùng **Histogram of Oriented Gradients (HOG)** để mô tả hướng cạnh.  
   - Kết hợp với **SVM** để phân loại vùng ảnh có phải khuôn mặt hay không.  
   - Chính xác hơn Viola–Jones.  
   - Nhược điểm: ít linh hoạt, chậm.  
   - Không phải deep learning.  

---

### Thuật toán Deep Learning methods (từ 2014 đến nay)  

1. **MTCNN (Multi-task Cascaded CNN, 2016)**  
   - Gồm 3 mạng CNN nhỏ:  
     - **P-Net (Proposal Network)** → đề xuất vùng có thể là mặt.  
     - **R-Net (Refine Network)** → loại bỏ vùng sai.  
     - **O-Net (Output Network)** → xác định bounding box + landmark (mắt, mũi, miệng).  
   - Rất phổ biến trong nhận diện khuôn mặt.  

2. **Faster R-CNN cho Face Detection**  
   - Sử dụng **Region Proposal Network (RPN)** để tìm vùng mặt.  
   - Chính xác cao, thường dùng trong hệ thống an ninh.  
   - Tốc độ chậm hơn YOLO/SSD.  

3. **SSD & YOLO (2016–nay)**  
   - Dùng one-stage detector, được huấn luyện riêng cho khuôn mặt.  
   - Rất nhanh, phù hợp real-time (camera, điện thoại).  
   - Đôi khi kém chính xác hơn RetinaFace.  
## 3. Thuận toán viola-jones 
 #### 1. Ý tưởng thuận toán : 
     + Giống con người khi nhìn một khuôn mặt của người khác họ sẽ tập trung vào những đặc điểm trên mặt vd : như mắt , mũi , miệng , gò má ... Sau nhiều lần quan sát chúng ta sẽ ghi nhớ một mặt người sẽ có những gì và mắt, mũi , miệng này có cấu trúc như thế nào để được xem là mặt người
     + Tuy nhiên máy tính thì lại không nhìn được vậy làm sao để nó nhận diện những gì có trong bức ảnh đây là lúc **haar-like** ra đời 
     
---

 #### 2. haar feature là gì : 
     + Là các bộ lọc trắng–đen (giống mắt, sống mũi, vùng má...).  
     + So sánh sự khác biệt độ sáng giữa các vùng.  
     + Dùng để mô tả đặc trưng khuôn mặt.  
     + Các haar feature sẽ bằng tổng pixel trong vùng tối - tổng cho vùng sáng
     + ảnh minh họa :
![alt text](https://user-images.githubusercontent.com/33037020/202063850-62ed2da9-1ac1-471b-a006-fa932b5c29a6.PNG) 
---

#### 3. Integral Image (Ảnh tích phân)   
+ Dùng để tăng tốc tính toán giá trị Haar feature.  
+ Mỗi điểm lưu tổng giá trị pixel từ góc trên–trái đến điểm đó.  
+ Nhờ đó, tính tổng vùng bất kỳ chỉ mất O(1).  

---

#### 4. AdaBoost  
+ Có hàng chục ngàn Haar features → cần chọn ra **những đặc trưng quan trọng nhất**.  (cơ chế này khá giống con người như đã giải thích ở trên) vậy chọn ra haar feature nào quan trọng như thế nào ?
  - Xét 1 haar feature bất kỳ có trong tập feature 
  - Duyệt hết haar feature này có trong tất cả các ảnh có mặt người hoặc không
  - Chọn ra ngưỡng Haar feature tốt nhất mà nó có thể phân chia giữa face và non-face 
  - Tính hàm error và lặp lại các bước trên 
![Anh](https://www.researchgate.net/profile/Mahdi-Rezaei-14/publication/258374050/figure/fig12/AS:668504957136915@1536395410760/Haar-feature-matching-inside-the-weak-classifiers.jpg)
+ AdaBoost sẽ:  
  - Kết hợp nhiều “weak classifiers” (phân loại yếu) thành “strong classifier”.  
  - Giúp giảm số đặc trưng cần dùng, tăng độ chính xác.  
  - Công thức AdaBoost trong Viola–Jones   : 

    1. Ban đầu, gán trọng số cho mỗi mẫu huấn luyện:  
       \[
       w_i = \frac{1}{N}, \quad i = 1, 2, ..., N
       \]

    2. Với mỗi vòng lặp t:  
       - Huấn luyện một **weak classifier** \( h_t(x) \) trên dữ liệu có trọng số \( w_i \).  
       - Tính **lỗi phân loại có trọng số**:  
         \[
         \varepsilon_t = \frac{\sum_{i=1}^N w_i \cdot \mathbf{}(h_t(x_i) \neq y_i)}{\sum_{i=1}^N w_i}
         \]
       - Tính **hệ số α** (độ tin cậy của weak classifier):  
         \[
         \alpha_t = \frac{1}{2} \ln \left( \frac{1 - \varepsilon_t}{\varepsilon_t} \right)
         \]
       - Cập nhật trọng số mẫu:  
         \[
         w_i \leftarrow w_i \cdot e^{-\alpha_t y_i h_t(x_i)}
         \]
         (nếu phân loại sai thì trọng số tăng, phân loại đúng thì giảm).  
       - Chuẩn hóa lại \( w_i \) để tổng = 1.  

    3. Bộ phân loại mạnh cuối cùng:  
       \[
       H(x) = \text{sign} \left( \sum_{t=1}^T \alpha_t h_t(x) \right)
       \]

---

#### 5. Cascade Classifier  
+ Các bộ phân loại mạnh được xếp theo **tầng (cascade)**.  
+ Vùng ảnh đi qua từng tầng kiểm tra:  
  - Nếu không đạt ở tầng nào → loại bỏ ngay.  
  - Nếu vượt qua tất cả tầng → được coi là khuôn mặt.  
+ Nhờ cách này, tốc độ phát hiện rất nhanh, phù hợp real-time.  

   

