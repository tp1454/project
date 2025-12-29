# 2D Molecular Descriptors

Trong phần này, mình tập trung giải quyết bài toán dự đoán nhiệt độ nóng chảy (Tm) sử dụng các **đặc trưng cấu trúc 2D (2D Molecular Descriptors)**.

* **Tại sao chọn 2D thay vì 3D?**

Trong nghiên cứu QSPR về nhiệt độ nóng chảy, việc lựa chọn không gian đặc trưng (Descriptor Space) đóng vai trò quyết định. Mình ưu tiên sử dụng các Đặc trưng 2D (2D Descriptors) thay vì 3D dựa trên các cơ sở học thuật sau:

Mặc dù $T_m$ là tính chất vật lý phụ thuộc vào cấu trúc tinh thể 3D, nhưng việc sử dụng descriptors 3D gặp phải vấn đề lớn về sự linh hoạt cấu trúc (conformational flexibility) và độ nhiễu cao. Các nghiên cứu của **Tetko et al. [[1]](#ref1)** và **Hughes et al. [[4]](#ref4)** đã chứng minh rằng các mô hình dựa trên 2D Descriptors thường đạt độ chính xác tương đương hoặc tốt hơn 3D nhờ tính ổn định (robustness) và khả năng tổng quát hóa cao hơn. Do đó, nghiên cứu này tối ưu hóa triệt để thông tin từ dữ liệu 2D thông qua các kỹ thuật chọn lọc đặc trưng tiên tiến, thay vì phức tạp hóa vấn đề bằng dữ liệu 3D.

* **Tính bất biến cấu hình (Conformational Invariance):** Các đặc trưng 3D phụ thuộc lớn vào cấu trúc không gian của phân tử. Tuy nhiên, một phân tử có thể tồn tại ở hàng ngàn cấu hình (conformers) khác nhau và việc xác định chính xác cấu hình "kết tinh" (crystal packing conformation) là cực kỳ tốn kém và dễ sai số. Theo **Karthikeyan et al. [[3]](#ref3)**, việc sử dụng các đặc trưng 2D (như chỉ số topo, fingerprints) mang lại độ ổn định cao hơn và tránh được nhiễu do việc tối ưu hóa hình học 3D không chính xác.

* **Hiệu suất tương đương:** Các nghiên cứu tổng quan của **Dearden [[2]](#ref2)** chỉ ra rằng mặc dù nhiệt độ nóng chảy là một tính chất phụ thuộc vào mạng tinh thể (3D), nhưng các mô hình dựa trên đặc trưng 2D vẫn đạt được độ chính xác ngang bằng hoặc thậm chí tốt hơn do loại bỏ được nhiễu, đặc biệt là trong các bài toán sàng lọc quy mô lớn (High-throughput screening).

# MODEL TRAINING: FEATURES SELECTION USING RFECV, GA

Trong bối cảnh bài toán Hóa học tính toán (Chemoinformatics) với không gian đặc trưng ban đầu lớn (937 descriptors), việc áp dụng các kỹ thuật giảm chiều dữ liệu là bắt buộc để giải quyết vấn đề Curse of Dimensionality và giảm thiểu hiện tượng quá khớp (Overfitting). Mình sử dụng tiếp cận lai (Hybrid approach) kết hợp hai thuật toán: RFECV (đại diện cho phương pháp Wrapper - Greedy) và Genetic Algorithm (đại diện cho phương pháp Stochastic/Heuristic).

## RFECV

**RFECV (Loại bỏ đặc trưng đệ quy có kiểm định chéo)**

Để giải quyết vấn đề số chiều lớn (High-dimensionality), mình áp dụng thuật toán Loại bỏ đặc trưng đệ quy tích hợp kiểm định chéo (RFECV). Đây là phương pháp được **Guyon et al. [[5]](#ref5)** đề xuất và được chứng minh là một trong những kỹ thuật mạnh mẽ nhất để loại bỏ các biến dư thừa (redundant features) mà không làm mất thông tin quan trọng.

Khác với các phương pháp lọc đơn giản (Filter methods), RFECV tương tác trực tiếp với mô hình (Wrapper method) để đánh giá tập hợp biến. Nghiên cứu của **Granitto et al. [[6]](#ref6)** trong lĩnh vực hóa trắc lượng (Chemometrics) cũng khẳng định rằng RFECV giúp giảm thiểu hiện tượng quá khớp (overfitting) và tăng khả năng tổng quát hóa của mô hình tốt hơn so với việc chỉ sử dụng Genetic Algorithm đơn lẻ.

1.2. Cơ chế hoạt động (Mechanism)

RFECV hoạt động dựa trên nguyên lý **loại bỏ lùi (Backward Elimination)** kết hợp với đánh giá chéo (Cross-Validation) để tìm số lượng đặc trưng tối ưu (). Quy trình cụ thể như sau:

1. **Huấn luyện khởi tạo:** Mô hình (trong trường hợp này là LightGBM) được huấn luyện trên toàn bộ tập đặc trưng ban đầu ().
2. **Xếp hạng (Ranking):** Thuật toán trích xuất tầm quan trọng của từng đặc trưng (feature importance) thông qua thuộc tính `feature_importances_` (dựa trên Gain hoặc Split trong cây quyết định).
3. **Loại bỏ (Pruning):** Các đặc trưng có trọng số thấp nhất sẽ bị loại bỏ khỏi tập dữ liệu.
4. **Lặp lại (Iteration):** Quá trình 1-3 được lặp lại cho đến khi tập đặc trưng rỗng.
5. **Kiểm định chéo (Cross-Validation):** Tại mỗi bước lặp, thuật toán ghi nhận điểm số (Scoring metric, ví dụ: RMSE hoặc R2) trên tập validation. Số lượng đặc trưng tối ưu () được xác định tại điểm mà sai số mô hình là thấp nhất.

1.3. Tại sao sử dụng RFECV?

* **Tính ổn định:** Việc tích hợp Cross-Validation (CV) giúp RFECV không bị chệch (bias) bởi một tập dữ liệu cụ thể, đảm bảo tập đặc trưng được chọn có khả năng tổng quát hóa cao.
* **Giải quyết Đa cộng tuyến:** RFECV có khả năng loại bỏ các đặc trưng dư thừa (redundant) có độ tương quan cao, giữ lại đặc trưng đại diện tốt nhất cho mô hình.

## Genetic Algorithm (GA)

2.1. Nguồn gốc và Định nghĩa

GA là một kỹ thuật tối ưu hóa ngẫu nhiên (stochastic optimization) dựa trên nguyên lý chọn lọc tự nhiên và di truyền học của Darwin, được **John Holland** giới thiệu lần đầu vào năm 1975 trong tác phẩm *"Adaptation in Natural and Artificial Systems"*. Trong Hóa học, GA được xem là phương pháp tiêu chuẩn để giải quyết các bài toán QSAR/QSPR phức tạp **(từ Leardi (2001)[[8]](#ref8))**.

2.2. Cơ chế hoạt động (Mechanism)

GA coi mỗi tập con đặc trưng là một "cá thể" (individual) và tập hợp các cá thể là một "quần thể" (population). Quá trình tiến hóa diễn ra qua các bước:

1. **Khởi tạo (Initialization):** Tạo ngẫu nhiên một quần thể các chuỗi nhị phân (binary strings/chromosomes), trong đó bit 1 đại diện cho việc chọn đặc trưng, bit 0 là không chọn.
2. **Đánh giá (Fitness Evaluation):** Mỗi cá thể được đánh giá bằng một hàm mục tiêu (Fitness function) – ở đây là giá trị RMSE của mô hình LightGBM khi sử dụng tập đặc trưng tương ứng.
3. **Chọn lọc (Selection):** Các cá thể có độ thích nghi tốt (RMSE thấp) được giữ lại để làm cha mẹ cho thế hệ sau (ví dụ: phương pháp Tournament Selection).
4. **Lai ghép (Crossover):** Trao đổi thông tin di truyền giữa các cặp cha mẹ để tạo ra cá thể con mới, nhằm kết hợp các đặc trưng tốt từ cả hai.
5. **Đột biến (Mutation):** Đảo ngẫu nhiên một số bit (0 thành 1 hoặc ngược lại) với xác suất thấp. Bước này cực kỳ quan trọng để duy trì sự đa dạng di truyền, giúp thuật toán thoát khỏi các điểm tối ưu cục bộ (Local Optima).

2.3. Tại sao sử dụng GA?

* **Tìm kiếm toàn cục (Global Search):** Khác với RFECV (vốn có tính chất tham lam - greedy), GA có khả năng duyệt không gian tìm kiếm rộng lớn và tìm ra nghiệm tối ưu toàn cục.
* **Phát hiện tương tác phi tuyến:** GA cực kỳ hiệu quả trong việc tìm ra các tổ hợp đặc trưng (synergistic features) mà khi đứng riêng lẻ thì không quan trọng, nhưng khi kết hợp lại thì cho hiệu quả dự đoán cao.

# UNION AND INTERSECTION 2 MODEL TRAINNING

**Giới thiệu sơ bộ về 2 phương pháp (Union và Intersection)**

Việc kết hợp nhiều thuật toán lựa chọn đặc trưng (Ensemble Feature Selection) được ***Saeys et al. [[9]](#ref9)*** đề xuất nhằm khắc phục tính không ổn định (instability) vốn có của các phương pháp đơn lẻ. Trong khi GA là thuật toán ngẫu nhiên có khả năng tìm kiếm toàn cục (Global Search), thì RFECV là thuật toán tất định thiên về tìm kiếm cục bộ (Local Search). Sự kết hợp này tận dụng được ưu điểm bổ trợ của cả hai, giúp mô hình tiếp cận được tập đặc trưng tối ưu hơn theo lý thuyết của ***Bolón-Canedo et al. [[10]](#ref10)***.

Để khai thác tối đa sức mạnh của cả RFECV (thống kê) và GA (tiến hóa), mình không chọn riêng lẻ một thuật toán mà áp dụng kỹ thuật Ensemble Feature Selection thông qua hai chiến lược:

**1. Phương pháp UNION (HỢP - $\cup$)**

Cơ chế: Kết hợp tất cả các đặc trưng được ít nhất một thuật toán (RFECV hoặc GA) lựa chọn.

$$S_{Union} = S_{RFECV} \cup S_{GA}$$

Mục tiêu: Mục tiêu là tối đa hóa không gian thông tin. Theo ***Guyon & Elisseeff [[11]](#ref11)***, các thuật toán lọc khác nhau thường có các 'điểm mù' (blind spots) khác nhau. Chiến lược Hợp (Union) giúp giảm thiểu rủi ro loại bỏ nhầm (False Negatives) các biến quan trọng mà một thuật toán đơn lẻ có thể bỏ qua do bias của thuật toán đó.

Đặc điểm: Tập đặc trưng lớn (727 features), giàu thông tin nhưng có rủi ro nhiễu cao.

**2. Phương pháp INTERSECTION (GIAO - $\cap$)**

Cơ chế: Chỉ giữ lại những đặc trưng được CẢ HAI thuật toán đồng thuận lựa chọn. Đây là chiến lược dựa trên sự đồng thuận (Consensus Approach). Theo nghiên cứu của ***Yang et al. [[12]](#ref12)***, các đặc trưng được lựa chọn bởi nhiều phương pháp khác nhau thường là các đặc trưng có tín hiệu mạnh (Strong signal) và ít phụ thuộc vào dữ liệu huấn luyện cụ thể, do đó mang lại tính ổn định (Robustness) cao nhất.

$$S_{Intersect} = S_{RFECV} \cap S_{GA}$$

Mục tiêu: Lọc nhiễu triệt để, chỉ giữ lại các đặc trưng "cốt lõi" (Robust Features) có độ tin cậy cao nhất về mặt thống kê và tương tác hóa học.

Đặc điểm: Tập đặc trưng gọn nhẹ (283 features), giảm thiểu Overfitting và tăng tốc độ mô hình.

## Features Union

Xây dựng một mô hình LightGBM sử dụng tập đặc trưng rộng lớn (khoảng 727 features) thu được từ phép hợp của RFECV và GA. Mục đích là để kiểm thử xem việc giữ lại nhiều thông tin có giúp mô hình dự đoán tốt hơn không.

Phương pháp Kiểm thử (Evaluation Methodology)

Quá trình đánh giá được thực hiện trên tập kiểm thử độc lập ($N_{test} = 1713$ mẫu) để đảm bảo tính khách quan. Quy trình bao gồm ba bước xử lý hậu kỳ (post-processing) quan trọng:

1. **Khôi phục Mô hình (Model Restoration):** Tải trọng số mô hình LightGBM và danh sách đặc trưng hợp nhất ($S_{Union}$) từ bộ nhớ.
2. **Biến đổi ngược thang đo (Inverse Transformation):**
Do mô hình được huấn luyện trên không gian Logarit ($y_{log} = \ln(1 + T_m)$) để giảm thiểu tác động của phân phối lệch, các giá trị dự đoán ($\hat{y}_{log}$) cần được chuyển đổi ngược về thang đo nhiệt độ thực (Kelvin) để có ý nghĩa vật lý:

$$\hat{y}_{real} = \exp(\hat{y}_{log}) - 1$$

Đoạn mã tích hợp cơ chế kiểm tra tự động (`threshold > 15`) để xác định xem dữ liệu đầu vào đang ở dạng Log hay dạng Thực, đảm bảo tính nhất quán khi tính toán sai số.

3. **Tính toán Chỉ số (Metric Calculation):** Sử dụng ba chỉ số tiêu chuẩn: MAE (Sai số tuyệt đối trung bình), RMSE (Căn bậc hai sai số trung bình bình phương) và  (Hệ số xác định).

**Kết luận**

Chiến lược Union hoạt động dựa trên nguyên lý không bỏ sót, chấp nhận giữ lại tất cả các đặc trưng tiềm năng mà bất kỳ thuật toán thành phần nào (RFE hoặc GA) đề xuất.

* Ưu điểm: Kết quả $R^2 > 0.63$ chứng minh rằng việc tổng hợp "trí tuệ" của nhiều thuật toán chọn lọc giúp tạo ra một tập dữ liệu đầu vào giàu thông tin, đảm bảo mô hình có đủ dữ kiện để học các mối quan hệ phức tạp.

* Hạn chế tồn tại: Mặc dù sở hữu số lượng đặc trưng lớn nhất, hiệu suất mô hình không tăng trưởng tuyến tính tương ứng. Điều này chỉ ra hiện tượng bão hòa thông tin (information saturation) - tức là việc thêm quá nhiều biến số không còn giúp giảm sai số đáng kể, mà ngược lại có thể làm tăng độ phức tạp tính toán và nguy cơ nhiễu.

## Features Intersection

**Thay đổi:** Thay vì nạp vào 727 features hỗn tạp (Union), mình chỉ nạp vào **283 features** nằm trong tập giao thoa.
* **Mục tiêu mới:** Chuyển từ chiến thuật "Lấy số lượng bù chất lượng" sang chiến thuật **"Tinh gọn và Chính xác"**. Mình muốn kiểm chứng giả thuyết: *Liệu một mô hình nhỏ gọn, sạch nhiễu có thể đánh bại một mô hình khổng lồ nhưng chứa nhiều "rác" hay không?*

* Phương pháp Kiểm thử (Evaluation Methodology)

Quá trình đánh giá được thực hiện trên tập kiểm thử độc lập ( mẫu), tập trung vào việc kiểm chứng giả thuyết về **tính tối giản (Model Parsimony)**:

1. **Dự báo (Prediction):** Mô hình LightGBM sử dụng tập đặc trưng giao thoa ( features) để dự báo nhiệt độ trên không gian Logarit.
2. **Hậu xử lý (Post-processing):** Kết quả dự báo () được chuyển đổi ngược về thang đo nhiệt độ thực (Kelvin) thông qua hàm mũ: .
3. **Đánh giá (Metrics):** Các chỉ số hiệu năng được tính toán dựa trên độ lệch giữa giá trị dự báo sau chuyển đổi và giá trị thực nghiệm.

Độ chính xác giải thích ($R^2$): 0.6298.Điều này cho thấy mô hình có khả năng nắm bắt được xấp xỉ 63% quy luật biến thiên của nhiệt độ nóng chảy dựa trên cấu trúc 2D. Đáng chú ý là dù số lượng đặc trưng giảm đi đáng kể (chỉ còn ~39% so với Union), khả năng giải thích dữ liệu của mô hình vẫn được bảo toàn gần như nguyên vẹn.

Sai số dự báo (Error Metrics):Với MAE $\approx$ 63.6 K và RMSE $\approx$ 96.2 K, sai số của mô hình nằm trong ngưỡng chấp nhận được đối với bài toán QSPR phức tạp này. Việc sai số chỉ tăng rất nhẹ (không đáng kể) so với mô hình Union chứng tỏ rằng các đặc trưng bị loại bỏ chủ yếu là nhiễu, và mô hình Intersection đã thành công trong việc tối ưu hóa giữa độ chính xác và sự tinh gọn.

Khi đối chiếu với mô hình Union, kết quả của Intersection cho thấy một sự đánh đổi thú vị giữa **độ chính xác** và **độ phức tạp**:

* **Hiệu năng tương đương (Comparable Performance):** Mức giảm của  là cực kỳ nhỏ (chỉ ~0.0015) và mức tăng sai số MAE không đáng kể (chỉ ~0.3 K). Điều này cho thấy mô hình Intersection vẫn duy trì được sức mạnh dự báo gần như nguyên vẹn.
* **Tối ưu hóa tài nguyên (Resource Optimization):** Quan trọng hơn, mô hình này đạt được kết quả trên với số lượng đặc trưng chỉ bằng **~39%** so với mô hình Union (283 vs 727 features).
* **Kết luận:** Chiến lược Intersection đã chứng minh được tính hiệu quả cao trong việc **loại bỏ dư thừa (Redundancy Reduction)**. Việc loại bỏ hơn 400 features mà không làm sụt giảm đáng kể độ chính xác chứng tỏ những features đó chủ yếu là nhiễu hoặc đa cộng tuyến. Mô hình Intersection do đó được đánh giá là gọn nhẹ, ổn định và ít rủi ro Overfitting hơn. Kết quả này phù hợp với nguyên lý Occam's Razor và định luật về sự tiết kiệm (Parsimony) trong thống kê được ***Hawkins [[13]](#ref13)*** nhấn mạnh: 'Trong hai mô hình có khả năng dự báo tương đương, mô hình đơn giản hơn luôn được ưu tiên'. Mô hình Intersection không chỉ giảm thiểu rủi ro quá khớp (Overfitting) mà còn tăng khả năng diễn giải hóa học (Chemical Interpretability)

# TÀI LIỆU THAM KHẢO (REFERENCES)

Các phương pháp và ngưỡng xử lý trong báo cáo này được tham chiếu dựa trên các công bố khoa học tiêu chuẩn trong lĩnh vực Hóa tin (Cheminformatics):

<a id="ref1"></a>
**[1]** Tetko, I. V., Tanchuk, V. Y., Kasheva, T. N., & Villa, A. E. (2001). Estimation of aqueous solubility of chemical compounds using E-state indices. *Journal of Chemical Information and Computer Sciences*, 41(6), 1488–1493. https://doi.org/10.1021/ci000392t
*(Công trình này chỉ ra rằng các mô tả 2D hoạt động tốt cho các tính chất phân tử độc lập, nhưng gặp hạn chế với các tính chất phụ thuộc vào mạng tinh thể như nhiệt độ nóng chảy).*

<a id="ref2"></a>
**[2]** Dearden, J. C. (1999). Quantitative structure-property relationships for prediction of boiling point, vapor pressure, and melting point. *Environmental Toxicology and Chemistry*, 22(8), 1696–1709. https://doi.org/10.1897/01-363
*(Paper kinh điển giải thích tại sao MP khó dự đoán và vai trò của 2D).*

<a id="ref3"></a>
**[3]** Karthikeyan, M., Glen, R. C., & Bender, A. (2005). General melting point prediction based on a diverse compound data set and artificial neural networks. *Journal of Chemical Information and Modeling*, 45(3), 581–590. https://doi.org/10.1021/ci0500132
*(Paper này sử dụng chủ yếu 2D descriptors cho mô hình MP lớn, chứng minh 2D là đủ).*

<a id="ref4"></a>
**[4]** Hughes, L. D., Palmer, D. S., Nigsch, F., & Mitchell, J. B. (2008). Why are some properties more difficult to predict than others? A study of QSPR models of solubility, melting point, and Log P. *Journal of Chemical Information and Modeling*, 48(1), 220–232. https://doi.org/10.1021/ci700307p
*(Tham chiếu cho việc loại bỏ các muối/chất vô cơ và giới hạn ngưỡng nhiệt độ < 1000K để tránh nhiễu).*

<a id="ref5"></a>
**[5]** Guyon, I., Weston, J., Barnhill, S., et al. (2002). Gene Selection for Cancer Classification using Support Vector Machines. *Machine Learning*, 46, 389–422. https://doi.org/10.1023/A:1012487302797
*(Cơ sở lý thuyết cho việc sử dụng RFE để loại bỏ lùi các đặc trưng yếu).*

<a id="ref6"></a>
**[6]** Granitto, P. M., Furlanello, C., Biasioli, F., & Gasperi, F. (2006). Recursive feature elimination with random forest for PTR-MS analysis of agroindustrial products. *Chemometrics and Intelligent Laboratory Systems*, 83(2), 83-90. https://doi.org/10.1016/J.CHEMOLAB.2006.01.007
*(Chứng minh hiệu quả của RFE trong dữ liệu hóa học thực nghiệm).*

<a id="ref7"></a>
**[7]** Tropsha, A., Gramatica, P., & Gombar, V. (2003). The Importance of Being Earnest: Validation is the Absolute Essential for Successful Application and Interpretation of QSPR Models. *QSAR & Combinatorial Science*, 22(1), 69-77. https://doi.org/10.1002/qsar.200390007
*(Tiêu chuẩn vàng để đánh giá độ tin cậy của một mô hình dự báo hóa học).*

<a id="ref8"></a>
**[8]** Leardi, R. (2001). Genetic algorithms in chemometrics and chemistry: a review. *Journal of Chemometrics*, 15(7), 559–569. https://doi.org/10.1002/cem.651
*(Bài báo tổng quan khẳng định vai trò của GA như một công cụ mạnh mẽ trong Hóa trắc lượng).*

<a id="ref9"></a> 
**[9]** Saeys, Y., Inza, I., & Larrañaga, P. (2007). A review of feature selection techniques in bioinformatics. Bioinformatics, 23(19), 2507–2517. https://doi.org/10.1093/bioinformatics/btm344 
*(Paper kinh điển về lý do nên dùng Ensemble Feature Selection để tránh bias).*

<a id="ref10"></a> 
**[10]** Bolón-Canedo, V., Sánchez-Maroño, N., & Alonso-Betanzos, A. (2012). Stability of feature selection algorithms. Artificial Intelligence Review, 37(3), 209-235. https://doi.org/10.1007/978-981-19-0151-5_26
*(Chứng minh rằng kết hợp nhiều thuật toán sẽ ổn định hơn việc chỉ chạy một thuật toán duy nhất).*

<a id="ref11"></a> 
**[11]** Guyon, I., & Elisseeff, A. (2003). An introduction to variable and feature selection. Journal of Machine Learning Research, 3, 1157-1182. https://doi.org/10.1162/153244303322753616
*(Cơ sở lý thuyết về bias và các điểm mù của các thuật toán chọn lọc đơn lẻ).*

<a id="ref12"></a> 
**[12]** Jian-Bo Yang and Chong-Jin Ong. 2010. Feature selection for support vector regression using probabilistic prediction. In Proceedings of the 16th ACM SIGKDD international conference on Knowledge discovery and data mining (KDD '10). Association for Computing Machinery, New York, NY, USA, 343–352. https://doi.org/10.1145/1835804.1835849
*(Nói về sức mạnh của phương pháp đồng thuận/giao thoa trong việc chọn ra các đặc trưng mạnh nhất).*

<a id="ref13"></a> 
**[13]** Hawkins D. M. (2004). The problem of overfitting. Journal of chemical information and computer sciences, 44(1), 1–12. https://doi.org/10.1021/ci0342472
*(Paper cực hay trong ngành Hóa tin về nguyên lý Parsimony: tại sao ít features lại tốt hơn cho khả năng diễn giải và tổng quát hóa).*