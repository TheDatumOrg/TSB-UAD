# List of methods

We provide here a list of anomaly detection methods proposed in the literature.

* Notation:
  I: Univariate, M: Multivariate; 
  S: Supervised, Se: Semi-Supervised and U: Unsupervised.

## Distance-base methods

| Method | Second Level | Prototype | Dim | Method | Stream |
|--------|--------------|-----------|-----|--------|--------|
| KNN [Hawkins, 1980] | Proximity-based | Nearest Neighbor | M | U | × |
| KnorrSeq2 [Palshikar, 2005] | Proximity-based | Nearest Neighbor | M | U | × |
| LOF [Breunig et al., 2000] | Proximity-based | LOF | M | U | × |
| COF [Tang et al., 2002] | Proximity-based | LOF | M | U | × |
| LOCI [Papadimitriou et al., 2003] | Proximity-based | LOF | M | U | ✓ |
| ILOF [Pokrajac et al., 2007] | Proximity-based | LOF | M | U | ✓ |
| DILOF [Na et al., 2018] | Proximity-based | LOF | M | U | ✓ |
| HSDE [Li et al., 2017] | Proximity-based | LOF | I | U | × |
| k-Means [Hawkins, 1980] | Clustering-based | k-Means | M | U | × |
| Hybrid-k-Means [Song et al., 2017] | Clustering-based | k-Means | M | U | × |
| DeepkMeans [Moradi Fard et al., 2020] | Clustering-based | k-Means | M | Se | × |
| DBSCAN [Sander et al., 1998] | Clustering-based | DBSCAN | M | U | × |
| DBStream [Hahsler and Bolaos, 2016] | Clustering-based | DBSCAN | M | U | ✓ |
| MCOD [Kontaki et al., 2011] | Clustering-based | - | I | U | × |
| CBLOF [He et al., 2003] | Clustering-based | LOF | M | U | × |
| sequenceMiner [Budalakoti et al., 2008] | Clustering-based | - | I | U | × |
| NorM (SAD) [Boniol et al., 2020] | Clustering-based | NormA | I | U | × |
| NormA [Boniol et al., 2021a] | Clustering-based | NormA | I | U | × |
| SAND [Boniol et al., 2021b] | Clustering-based | NormA | I | U | ✓ |
| TARZAN[Keogh et al., 2002] | Discord-based | - | I | S | × |
| HOT SAX [Keogh et al., 2005] | Discord-based | - | I | U | × |
| DAD [Yankov et al., 2008] | Discord-based | - | I | U | × |
| AMD [Yang and Liao, 2017] | Discord-based | - | I | U | × |
| STAMPI [Yeh et al., 2016] | Discord-based | Matrix Profile | M | U | ✓ |
| STOMP [Zhu et al., 2016] | Discord-based | Matrix Profile | M | U | × |
| MERLIN [Nakamura et al., 2020] | Discord-based | Matrix Profile | I | U | × |
| MERLIN++ [Nakamura et al., 2023] | Discord-based | Matrix Profile | I | U | × |
| SCRIMP [Zhu et al., 2018] | Discord-based | Matrix Profile | I | U | × |
| SCAMP [Zimmerman et al., 2019a] | Discord-based | Matrix Profile | I | U | × |
| VALMOD [Linardi et al., 2020] | Discord-based | Matrix Profile | I | U | ✓ |
| DAMP [Lu et al., 2022] | Discord-based | Matrix Profile | I | U | ✓ |
| LAMP [Zimmerman et al., 2019b] | Discord-based | Matrix Profile | I | Se | ✓ |


## Density-based Methods

| Method | Second Level | Prototype | Dim | Method | Stream |
|--------|--------------|-----------|-----|--------|--------|
| FAST-MCD [Rousseeuw and Driessen, 1999] | Distribution-based | MCD | M | Se | × |
| MC-MCD [Hardin and Rocke, 2004] | Distribution-based | MCD | M | Se | × |
| OCSVM [Ma and Perkins, 2003b] | Distribution-based | SVM | M | Se | × |
| AOSVM [Gomez-Verdejo et al., 2011] | Distribution-based | SVM | M | U | ✓ |
| Eros-SVMs [Lamrini et al., 2018] | Distribution-based | SVM | M | Se | × |
| S-SVM [Bhargava and Raghuvanshi, 2013] | Distribution-based | SVM | I | Se | × |
| MS-SVDD [Xiao et al., 2009] | Distribution-based | SVM | M | Se | × |
| NetworkSVM [Zhang et al., 2007] | Distribution-based | SVM | M | Se | × |
| HMAD [Gornitz et al., 2015] | Distribution-based | SVM | I | Se | × |
| DeepSVM [Wu et al., 2020] | Distribution-based | SVM | M | U | × |
| HBOS [Goldstein and Dengel, 2013] | Distribution-based | - | M | U | × |
| COPOD [Li et al., 2020] | Distribution-based | - | M | U | × |
| ConInd [Antoni and Borghesani, 2019] | Distribution-based | - | M | Se | × |
| MGDD [Subramaniam et al., 2006] | Distribution-based | - | M | U | ✓ |
| OC-KFD [Roth, 2006] | Distribution-based | - | M | U | × |
| SmartSifter [Yamanishi et al., 2004] | Distribution-based | - | M | U | ✓ |
| MedianMethod [Basu and Meckesheimer, 2007] | Distribution-based | - | I | U | ✓ |
| S-ESD [Hochenbaum et al., 2017] | Distribution-based | ESD | I | U | × |
| S-H-ESD [Hochenbaum et al., 2017] | Distribution-based | ESD | I | U | × |
| SH-ESD+ [Vieira et al., 2018] | Distribution-based | ESD | I | U | × |
| TwoFinger [Marceau, 2000] | Graph-based | - | I | Se | × |
| GeckoFSM [Salvador and Chan, 2005] | Graph-based | - | M | S | × |
| Series2Graph [Boniol and Palpanas, 2020] | Graph-based | Series2Graph | I | U | × |
| DADS [Schneider et al., 2021] | Graph-based | Series2Graph | I | U | × |
| IForest [Liu et al., 2008] | Tree-based | IForest | M | U | × |
| IF-LOF [Cheng et al., 2019] | Tree-based | IForest/LOF | M | U | × |
| Extended IForest [Hariri et al., 2019] | Tree-based | IForest | M | U | × |
| Hybrid IForest [Marteau et al., 2017] | Tree-based | IForest | M | Se | × |
| SurpriseEncode [Chakrabarti et al., 1998] | Encoding-based | - | M | U | × |
| GranmmarViz [Senin et al., 2015] | Encoding-based | - | I | U | × |
| Ensemble GI [Gao et al., 2020] | Encoding-based | - | I | U | × |
| PST [Sun et al., 2006] | Encoding-based | Markov Ch. | M | U | × |
| EM-HMM [Park et al., 2016] | Encoding-based | Markov Ch. | M | Se | ✓ |
| LaserDBN [Ogbechie et al., 2017] | Encoding-based | Bayesian Net. | M | Se | × |
| EDBN [Pauwels and Calders, 2019a] | Encoding-based | Bayesian Net. | M | Se | × |
| KDE-EDBN [Pauwels and Calders, 2019b] | Encoding-based | Bayesian Net. | M | Se | × |
| PCA [Snyder and Withers, 1983a] | Encoding-based | PCA | M | Se | × |
| RobustPCA [Paffenroth et al., 2018] | Encoding-based | PCA | M | U | × |
| DeepPCA [Chalapathy et al., 2017] | Encoding-based | PCA | M | Se | × |
| POLY [Yao et al., 2010] | Encoding-based | - | I | U | × |
| SSA [Yao et al., 2010] | Encoding-based | - | I | U | × |


## Prediction-based Methods

| Method | Second Level | Prototype | Dim | Method | Stream |
|--------|--------------|-----------|-----|--------|--------|
| ES [Snyder and Withers, 1983b] | Forecasting-based | - | I | Se | × |
| DES [Snyder and Withers, 1983b] | Forecasting-based | - | I | Se | × |
| TES [Snyder and Withers, 1983b] | Forecasting-based | - | I | U | × |
| ARIMA [Rousseeuw and Leroy, 1987] | Forecasting-based | ARIMA | I | U | ✓ |
| NoveltySVR [Ma and Perkins, 2003a] | Forecasting-based | SVM | I | U | ✓ |
| PCI [Yu et al., 2014] | Forecasting-based | ARIMA | I | U | ✓ |
| OceanWNN [Wang et al., 2019] | Forecasting-based | - | I | Se | × |
| MTAD-GAT [Zhao et al., 2020] | Forecasting-based | GRU | M | Se | ✓ |
| AD-LTI [Wu et al., 2020] | Forecasting-based | GRU | M | Se | ✓ |
| CoalESN [Obst et al., 2008] | Forecasting-based | ESN | M | Se | ✓ |
| MoteESN [Chang et al., 2009] | Forecasting-based | ESN | I | Se | ✓ |
| HealthESN [Chen et al., 2020] | Forecasting-based | ESN | I | Se | × |
| Torsk [Heim and Avery, 2019] | Forecasting-based | ESN | M | U | ✓ |
| LSTM-AD [Malhotra et al., 2015] | Forecasting-based | LSTM | M | Se | × |
| DeepLSTM [Chauhan and Vig, 2015] | Forecasting-based | LSTM | I | Se | × |
| DeepAnT [Munir et al., 2019] | Forecasting-based | LSTM | M | Se | × |
| Telemanom [Hundman et al., 2018] | Forecasting-based | LSTM | M | Se | × |
| RePAD [Lee et al., 2020] | Forecasting-based | LSTM | M | U | × |
| NumentaHTM [Ahmad et al., 2017] | Forecasting-based | HTM | I | U | ✓ |
| MultiHTM [Wu et al., 2018] | Forecasting-based | HTM | M | U | ✓ |
| RADM [Ding et al., 2018] | Forecasting-based | HTM | M | Se | ✓ |
| MAD-GAN [Li et al., 2019] | Reconstruction-based | GAN | M | Se | ✓ |
| VAE-GAN [Niu et al., 2020] | Reconstruction-based | GAN | M | Se | × |
| TAnoGAN [Bashar and Nayak, 2020] | Reconstruction-based | GAN | M | Se | × |
| USAD [Audibert et al., 2020] | Reconstruction-based | GAN | M | Se | × |
| EncDec-AD [Malhotra et al., 2016] | Reconstruction-based | AE | M | Se | × |
| LSTM-VAE [Park et al., 2018] | Reconstruction-based | AE | M | Se | ✓ |
| DONUT [Xu et al., 2018] | Reconstruction-based | AE | I | Se | × |
| BAGEL [Li et al., 2018] | Reconstruction-based | AE | I | Se | × |
| OmniAnomaly [Su et al., 2019] | Reconstruction-based | AE | M | Se | × |
| MSCRED [Zhang et al., 2019] | Reconstruction-based | AE | I | U | × |
| VELC [Zhang et al., 2020] | Reconstruction-based | AE | I | Se | × |
| CAE [Garcia et al., 2020] | Reconstruction-based | AE | I | Se | × |
| DeepNAP [Kim et al., 2018] | Reconstruction-based | AE | M | Se | ✓ |
| STORN [Soelch et al., 2016] | Reconstruction-based | AE | M | Se | ✓ |

# References

[Ahmad et al., 2017] Subutai Ahmad, Alexander Lavin, Scott Purdy, and Zuha Agha. Unsupervised real-time anomaly detection for streaming data. 262:134–147, 2017.

[Antoni and Borghesani, 2019] Jerome Antoni and Pietro Borghesani. A statistical methodology for the design of condition indicators. 114:290–327, 2019.

[Audibert et al., 2020] Julien Audibert, Pietro Michiardi, Fr´ed´eric
Guyard, S´ebastien Marti, and Maria A Zuluaga. Usad: Unsupervised
anomaly detection on multivariate time series. In SIGKDD, pages 3395–3404, 2020.

[Bashar and Nayak, 2020] Md Abul Bashar and Richi Nayak. Tanogan: Time series anomaly detection with generative adversarial networks. In SSCI, pages 1778–1785. IEEE, 2020.

[Basu and Meckesheimer, 2007] Sabyasachi Basu and Martin Meckesheimer. Automatic outlier detection for time series: An application to sensor data. 11(2):137–154, 2007.

[Bhargava and Raghuvanshi, 2013] Arpita Bhargava and AS Raghuvanshi. Anomaly detection in wireless sensor networks using s-transform in combination with svm. In 2013 5th International Conference and Computational Intelligence and Communication Networks, pages 111–116. IEEE, 2013.

[Boniol and Palpanas, 2020] Paul Boniol and Themis Palpanas. Series2graph: Graph-based subsequence anomaly detection for time series. PVLDB, 13(11), 2020.

[Boniol et al., 2020] Paul Boniol, Michele Linardi, Federico Roncallo, and Themis Palpanas. Automated anomaly detection in large sequences. In 2020 IEEE 36th international conference on data engineering (ICDE), pages 1834–1837. IEEE, 2020.

[Boniol et al., 2021a] Paul Boniol, Michele Linardi, Federico Roncallo, Themis Palpanas, Mohammed Meftah, and Emmanuel Remy. Unsupervised and scalable subsequence anomaly detection in large data series. The VLDB Journal, March 2021. 

[Boniol et al., 2021b] Paul Boniol, John Paparrizos, Themis Palpanas,
and Michael J Franklin. Sand: streaming subsequence anomaly detection. PVLDB, 14(10):1717–1729, 2021.

[Breunig et al., 2000] Markus M Breunig, Hans-Peter Kriegel,
Raymond T Ng, and J¨org Sander. Lof: identifying density-based local outliers. In Proceedings of the 2000 ACM SIGMOD international
conference on Management of data, pages 93–104,
2000.

[Budalakoti et al., 2008] Suratna Budalakoti, Ashok N Srivastava,
and Matthew Eric Otey. Anomaly detection and diagnosis algorithms
for discrete symbol sequences with applications to airline
safety. IEEE Transactions on Systems, Man, and Cybernetics,
Part C (Applications and Reviews), 39(1):101–113, 2008.

[Chakrabarti et al., 1998] Soumen Chakrabarti, Sunita Sarawagi,
and Byron Dom. Mining Surprising Patterns Using Temporal
Description Length. In Proceedings of the International Conference
on Very Large Databases (VLDB), volume 24 of VLDB ’98,
pages 606–617. Morgan Kaufmann Publishers Inc., 1998.

[Chalapathy et al., 2017] Raghavendra Chalapathy, Aditya Krishna
Menon, and Sanjay Chawla. Robust, deep and inductive anomaly
detection. In Michelangelo Ceci, Jaakko Hollm´en, Ljupˇco Todorovski,
Celine Vens, and Saˇso Dˇzeroski, editors, Machine Learning
and Knowledge Discovery in Databases, pages 36–51, Cham,
2017. Springer International Publishing.

[Chang et al., 2009] Marcus Chang, Andreas Terzis, and Philippe
Bonnet. Mote-Based Online Anomaly Detection Using Echo
State Networks. In Bhaskar Krishnamachari, Subhash Suri,
Wendi Heinzelman, and Urbashi Mitra, editors, Proceedings of
the International Conference on Distributed Computing in Sensor
Systems (DCOOS), volume 5516 of Lecture Notes in Computer
Science, pages 72–86. Springer Berlin Heidelberg, 2009.

[Chauhan and Vig, 2015] S. Chauhan and L. Vig. Anomaly detection
in ECG time signals via deep long short-term memory networks.
In Proceedings of the International Conference on Data
Science and Advanced Analytics (DSAA), pages 1–7, 2015.

[Chen et al., 2020] Qing Chen, Anguo Zhang, Tingwen Huang,
Qianping He, and Yongduan Song. Imbalanced dataset-based
echo state networks for anomaly detection. 32(8):3685–3694,
2020.

[Cheng et al., 2019] Zhangyu Cheng, Chengming Zou, and Jianwei Dong. Outlier detection using isolation forest and local outlier factor. In Proceedings of the conference on research in adaptive and convergent systems, pages 161–168, 2019.

[Ding et al., 2018] Nan Ding, Huanbo Gao, Hongyu Bu, Haoxuan Ma, and Huaiwei Si. Multivariate-Time-Series-Driven Real-time Anomaly Detection Based on Bayesian Network. 18(10):3367,
2018.

[Gao et al., 2020] Yifeng Gao, Jessica Lin, and Constantin Brif. Ensemble Grammar Induction For Detecting Anomalies in Time Series. In Proceedings of the International Conference on Extending Database Technology (EDBT), 2020.

[Garcia et al., 2020] Gabriel Garcia, Gabriel Michau, Melanie Ducoffe, Jayant Sen Gupta, and Olga Fink. Time series to assets: Monitoring the condition of industrial assets with deep learning image processing algorithms. 05 2020.

[Goldstein and Dengel, 2013] Markus Goldstein and Andreas Dengel. Histogram-based outlier score (hbos): A fast unsupervised anomaly detection algorithm, 2013.

[G´omez-Verdejo et al., 2011] Vanessa G´omez-Verdejo, Jer´onimo Arenas-Garc´ıa, Miguel Lazaro-Gredilla, and A´ ngel Navia-Vazquez. Adaptive one-class support vector machine. IEEE Transactions on Signal Processing, 59(6):2975–2981, 2011.

[G¨ornitz et al., 2015] Nico G¨ornitz, Mikio Braun, and Marius
Kloft. Hidden Markov anomaly detection. In Proceedings
of the International Conference on Machine Learning (ICML),
ICML’15, pages 1833–1842. JMLR.org, 2015.

[Hahsler and Bolaos, 2016] Michael Hahsler and Matthew Bolaos.
Clustering data streams based on shared density between
micro-clusters. IEEE Trans. on Knowl. and Data Eng.,
28(6):1449–1461, jun 2016.

[Hardin and Rocke, 2004] Johanna Hardin and David M Rocke.
Outlier detection in the multiple cluster setting using the minimum
covariance determinant estimator. Computational Statistics
& Data Analysis, 44(4):625 – 638, 2004.

[Hariri et al., 2019] Sahand Hariri, Matias Carrasco Kind, and
Robert J Brunner. Extended isolation forest. IEEE transactions
on knowledge and data engineering, 33(4):1479–1489, 2019.

[Hawkins, 1980] D. M Hawkins. Identification of Outliers.
Springer Netherlands, Dordrecht, 1980. OCLC: 945065134.
[He et al., 2003] Zengyou He, Xiaofei Xu, and Shengchun Deng.
Discovering cluster-based local outliers. Pattern recognition letters,
24(9-10):1641–1650, 2003.

[Heim and Avery, 2019] Niklas Heim and James E. Avery. Adaptive
Anomaly Detection in Chaotic Time Series with a Spatially
Aware Echo State Network, 2019.

[Hochenbaum et al., 2017] Jordan Hochenbaum, Owen S. Vallis,
and Arun Kejariwal. Automatic Anomaly Detection in the Cloud
Via Statistical Learning, 2017.

[Hundman et al., 2018] Kyle Hundman, Valentino Constantinou,
Christopher Laporte, Ian Colwell, and Tom Soderstrom. Detecting
Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic
Thresholding. In Proceedings of the International Conference
on Knowledge Discovery and Data Mining (SIGKDD),
pages 387–395. ACM, 2018.

[Keogh et al., 2002] Eamonn Keogh, Stefano Lonardi, and
Bill’Yuan-chi’ Chiu. Finding surprising patterns in a time
series database in linear time and space. In Proceedings of the
eighth ACM SIGKDD international conference on Knowledge
discovery and data mining, pages 550–556, 2002.

[Keogh et al., 2005] Eamonn Keogh, Jessica Lin, and Ada Fu. Hot
sax: Efficiently finding the most unusual time series subsequence.
In Fifth IEEE International Conference on Data Mining
(ICDM’05), pages 8–pp. Ieee, 2005.

[Kim et al., 2018] Chunggyeom Kim, Jinhyuk Lee, Raehyun Kim,
Youngbin Park, and Jaewoo Kang. DeepNAP: Deep neural
anomaly pre-detection in a semiconductor fab. 457-458:1–11,
2018.

[Kontaki et al., 2011] Maria Kontaki, Anastasios Gounaris, Apostolos
N Papadopoulos, Kostas Tsichlas, and Yannis Manolopoulos.
Continuous monitoring of distance-based outliers over data
streams. In 2011 IEEE 27th International Conference on Data
Engineering, pages 135–146. IEEE, 2011.

[Lamrini et al., 2018] Bouchra Lamrini, Augustin Gjini, Simon
Daudin, Pascal Pratmarty, Franc¸ois Armando, and Louise Trav´e-
Massuy`es. Anomaly detection using similarity-based one-class
svm for network traffic characterization. In DX, 2018.

[Lee et al., 2020] Ming-Chang Lee, Jia-Chun Lin, and Ernst Gunnar
Gran. RePAD: Real-Time Proactive Anomaly Detection
for Time Series. In Leonard Barolli, Flora Amato, Francesco
Moscato, Tomoya Enokido, and Makoto Takizawa, editors, Proceedings
of the International Conference on Advanced Information
Networking and Applications (AINA), Advances in Intelligent
Systems and Computing, pages 1291–1302. Springer International
Publishing, 2020.

[Li et al., 2017] Zhihua Li, Ziyuan Li, Ning Yu, Steven Wen, et al.
Locality-based visual outlier detection algorithm for time series.
Security and Communication Networks, 2017, 2017.

[Li et al., 2018] Zeyan Li, Wenxiao Chen, and Dan Pei. Robust
and Unsupervised KPI Anomaly Detection Based on Conditional
Variational Autoencoder. In Proceedings of the International
Performance Computing and Communications Conference
(IPCCC), pages 1–9. IEEE, 2018.

[Li et al., 2019] Dan Li, Dacheng Chen, Baihong Jin, Lei Shi,
Jonathan Goh, and See-Kiong Ng. Mad-gan: Multivariate
anomaly detection for time series data with generative adversarial
networks. In International conference on artificial neural networks,
pages 703–716. Springer, 2019.

[Li et al., 2020] Zheng Li, Yue Zhao, Nicola Botta, Cezar Ionescu,
and Xiyang Hu. COPOD: copula-based outlier detection. In
IEEE International Conference on Data Mining (ICDM). IEEE,
2020.

[Linardi et al., 2020] Michele Linardi, Yan Zhu, Themis Palpanas,
and Eamonn Keogh. Matrix profile goes mad: variable-length
motif and discord discovery in data series. Data Mining and
Knowledge Discovery, 34:1022–1071, 2020.

[Liu et al., 2008] Fei Tony Liu, Kai Ming Ting, and Zhi-Hua Zhou.
Isolation forest. In ICDM, pages 413–422. IEEE, 2008.

[Lu et al., 2022] Yue Lu, Renjie Wu, Abdullah Mueen, Maria A
Zuluaga, and Eamonn Keogh. Matrix profile xxiv: scaling time
series anomaly detection to trillions of datapoints and ultra-fast
arriving data streams. In SIGKDD, pages 1173–1182, 2022.

[Ma and Perkins, 2003a] Junshui Ma and Simon Perkins. Online
novelty detection on temporal sequences. In Proceedings of
the International Conference on Knowledge Discovery and Data
Mining (SIGKDD), page 613. ACM Press, 2003.

[Ma and Perkins, 2003b] Junshui Ma and Simon Perkins. Timeseries
novelty detection using one-class support vector machines.
In Proceedings of the International Joint Conference on Neural
Networks, 2003., volume 3, pages 1741–1745. IEEE, 2003.

[Malhotra et al., 2015] Pankaj Malhotra, Lovekesh Vig, Gautam
Shroff, and Puneet Agarwal. Long Short Term Memory Networks
for Anomaly Detection in Time Series. In Proceedings of
the European Symposium on Artificial Neural Networks, Computational
Intelligence and Machine Learning (ESANN), volume
23, 2015.

[Malhotra et al., 2016] Pankaj Malhotra, Anusha Ramakrishnan,
Gaurangi Anand, Lovekesh Vig, Puneet Agarwal, and Gautam
Shroff. LSTM-based Encoder-Decoder for Multi-sensor
Anomaly Detection, 2016.

[Marceau, 2000] Carla Marceau. Characterizing the behavior of a
program using multiple-length N-grams. In Proceedings of the
Workshop on New Security Paradigms (NSPW), pages 101–110.
ACM Press, 2000.

[Marteau et al., 2017] Pierre-Franc¸ois Marteau, Saeid Soheily-
Khah, and Nicolas B´echet. Hybrid isolation forest-application
to intrusion detection. arXiv preprint arXiv:1705.03800, 2017.

[Moradi Fard et al., 2020] Maziar Moradi Fard, Thibaut Thonet,
and Eric Gaussier. Deep k-means: Jointly clustering with kmeans
and learning representations. Pattern Recognition Letters,
138:185 – 192, 2020.

[Munir et al., 2019] Mohsin Munir, Shoaib Ahmed Siddiqui, Andreas
Dengel, and Sheraz Ahmed. DeepAnT: A Deep Learning
Approach for Unsupervised Anomaly Detection in Time Series.
7:1991–2005, 2019.

[Na et al., 2018] Gyoung S Na, Donghyun Kim, and Hwanjo Yu.
Dilof: Effective and memory efficient local outlier detection in
data streams. In SIGKDD, pages 1993–2002, 2018.
[Nakamura et al., 2020] Takaaki Nakamura, Makoto Imamura,
Ryan Mercer, and Eamonn J. Keogh. MERLIN: parameter-free
discovery of arbitrary length anomalies in massive time series
archives. In Claudia Plant, Haixun Wang, Alfredo Cuzzocrea,
Carlo Zaniolo, and Xindong Wu, editors, ICDM, pages 1190–
1195. IEEE, 2020.

[Nakamura et al., 2023] Takaaki Nakamura, Ryan Mercer, Makoto
Imamura, and Eamonn Keogh. Merlin++: parameter-free discovery
of time series anomalies. Data Mining and Knowledge
Discovery, 37(2):670–709, 2023.
[Niu et al., 2020] Zijian Niu, Ke Yu, and Xiaofei Wu. Lstm-based
vae-gan for time-series anomaly detection. Sensors, 20(13):3738,
2020.

[Obst et al., 2008] Oliver Obst, X. Rosalind Wang, and Mikhail
Prokopenko. Using Echo State Networks for Anomaly Detection
in Underground Coal Mines. In Proceedings of the International
Conference on Information Processing in Sensor Networks
(IPSN), pages 219–229. IEEE, 2008.

[Ogbechie et al., 2017] Alberto Ogbechie, Javier D´ıaz-Rozo, Pedro
Larra˜naga, and Concha Bielza. Dynamic Bayesian Network-
Based Anomaly Detection for In-Process Visual Inspection of
Laser Surface Heat Treatment. In J¨urgen Beyerer, Oliver Niggemann,
and Christian K¨uhnert, editors, Proceedings of the International
Conference on Machine Learning for Cyber Physical Systems
(ML4CPS), pages 17–24. Springer Berlin Heidelberg, 2017.

[Paffenroth et al., 2018] Randy Paffenroth, Kathleen Kay, and Les
Servi. Robust PCA for Anomaly Detection in Cyber Networks,
2018.

[Palshikar, 2005] Girish Keshav Palshikar. Distance-based outliers
in sequences. In Distributed Computing and Internet Technology:
Second International Conference, ICDCIT 2005, BhubaneswarIndia, December 22-24, 2005. Proceedings 2, pages 547–552.
Springer, 2005.

[Papadimitriou et al., 2003] Spiros Papadimitriou, Hiroyuki Kitagawa,
Phillip B Gibbons, and Christos Faloutsos. Loci: Fast
outlier detection using the local correlation integral. In ICDE,
pages 315–326. IEEE, 2003.

[Park et al., 2016] Daehyung Park, Zackory Erickson, Tapomayukh
Bhattacharjee, and Charles C. Kemp. Multimodal execution
monitoring for anomaly detection during robot manipulation. In
Proceedings of the International Conference on Robotics and Automation
(ICRA), pages 407–414. IEEE, 2016.

[Park et al., 2018] Daehyung Park, Yuuna Hoshi, and Charles C.
Kemp. A Multimodal Anomaly Detector for Robot-Assisted
Feeding Using an LSTM-Based Variational Autoencoder.
3(3):1544–1551, 2018.

[Pauwels and Calders, 2019a] Stephen Pauwels and Toon Calders.
An anomaly detection technique for business processes based
on extended dynamic bayesian networks. In Proceedings of the
ACM/SIGAPP Symposium on Applied Computing (SAC), pages
494–501. ACM, 2019.

[Pauwels and Calders, 2019b] Stephen Pauwels and Toon Calders.
Detecting anomalies in hybrid business process logs. 19(2):18–
30, 2019.

[Pokrajac et al., 2007] Dragoljub Pokrajac, Aleksandar Lazarevic,
and Longin Jan Latecki. Incremental local outlier detection for
data streams. In 2007 IEEE symposium on computational intelligence
and data mining, pages 504–515. IEEE, 2007.

[Roth, 2006] Volker Roth. Kernel Fisher Discriminants for Outlier
Detection. 18(4):942–960, 2006.

[Rousseeuw and Driessen, 1999] Peter J. Rousseeuw and Katrien
Van Driessen. A fast algorithm for the minimum covariance
determinant estimator. Technometrics, 41(3):212–223, 1999.

[Rousseeuw and Leroy, 1987] Peter J. Rousseeuw and Annick M.
Leroy. Robust regression and outlier detection. Wiley series in
probability and mathematical statistics. Wiley, New York, 1987.

[Salvador and Chan, 2005] Stan Salvador and Philip Chan. Learning
States and Rules for Detecting Anomalies in Time Series.
23(3):241–255, 2005.

[Sander et al., 1998] J¨org Sander, Martin Ester, Hans-Peter Kriegel,
and Xiaowei Xu. Density-based clustering in spatial databases:
The algorithm gdbscan and its applications. Data Mining and
Knowledge Discovery, 2(2):169–194, Jun 1998.

[Schneider et al., 2021] Johannes Schneider, Phillip Wenig, and
Thorsten Papenbrock. Distributed detection of sequential anomalies
in univariate time series. The VLDB Journal, 30(4):579–602,
mar 2021.

[Senin et al., 2015] Pavel Senin, Jessica Lin, Xing Wang, Tim
Oates, Sunil Gandhi, Arnold P. Boedihardjo, Crystal Chen,
and Susan Frankenstein. Time series anomaly discovery with
grammar-based compression, 2015.

[Snyder and Withers, 1983a] Ralph D. Snyder and Stephen J.Withers.
Exponential smoothing with finite sample correction.
(1983,1), 1983.

[Snyder and Withers, 1983b] Ralph D. Snyder and Stephen J.Withers.
Exponential smoothing with finite sample correction. Number
1983,1 in Working paper. Department of Econometrics and
Operations Research. Monash University. Dept., Univ, Clayton,
1983.

[Soelch et al., 2016] Maximilian Soelch, Justin Bayer, Marvin
Ludersdorfer, and Patrick van der Smagt. Variational Inference
for On-line Anomaly Detection in High-Dimensional Time Series,
2016.

[Song et al., 2017] Hongchao Song, Zhuqing Jiang, Aidong Men,
and Bo Yang. A hybrid semi-supervised anomaly detection
model for high-dimensional data. Computational Intelligence
and Neuroscience, 2017:8501683, Nov 2017.

[Su et al., 2019] Ya Su, Youjian Zhao, Chenhao Niu, Rong Liu,Wei
Sun, and Dan Pei. Robust Anomaly Detection for Multivariate
Time Series through Stochastic Recurrent Neural Network. In
SIGKDD, pages 2828–2837. ACM, 2019.

[Subramaniam et al., 2006] S. Subramaniam, T. Palpanas, D. Papadopoulos,
V. Kalogeraki, and D. Gunopulos. Online outlier detection
in sensor data using non-parametric models. In Proceedings
of the International Conference on Very Large Databases
(VLDB), VLDB ’06, pages 187–198. VLDB Endowment, 2006.

[Sun et al., 2006] Pei Sun, Sanjay Chawla, and Bavani
Arunasalam. Mining for Outliers in Sequential Databases.
In Proceedings of the International Conference on Data Mining
(ICDM), pages 94–105. Society for Industrial and Applied
Mathematics, 2006.

[Tang et al., 2002] Jian Tang, Zhixiang Chen, Ada Wai-Chee Fu,
and David W Cheung. Enhancing effectiveness of outlier detections
for low density patterns. In PAKDD, pages 535–548, 2002.

[Vieira et al., 2018] Rafael G. Vieira, Marcos A. Leone Filho, and
Robinson Semolini. An Enhanced Seasonal-Hybrid ESD Technique
for Robust Anomaly Detection on Time Series. In Simp´osio
Brasileiro de Redes de Computadores (SBRC), volume 36, 2018.

[Wang et al., 2019] YiWang, Linsheng Han,Wei Liu, Shujia Yang,
and Yanbo Gao. Study on wavelet neural network based anomaly
detection in ocean observing data series. 186:106129, 2019.

[Wu et al., 2018] Jia Wu, Weiru Zeng, and Fei Yan. Hierarchical
Temporal Memory method for time-series-based anomaly detection.
273:535–546, 2018.

[Wu et al., 2020] P. Wu, J. Liu, and F. Shen. A deep one-class neural
network for anomalous event detection in complex scenes.
IEEE Transactions on Neural Networks and Learning Systems,
31(7):2609–2622, 2020.

[Wu et al., 2020 10 29] WentaiWu, Ligang He,Weiwei Lin, Yi Su,
Yuhua Cui, Carsten Maple, and Stephen Jarvis. Developing an
Unsupervised Real-time Anomaly Detection Scheme for Time
Series with Multi-seasonality, 2020-10-29.

[Xiao et al., 2009] Yanshan Xiao, Bo Liu, Longbing Cao, Xindong
Wu, Chengqi Zhang, Zhifeng Hao, Fengzhao Yang, and Jie Cao.
Multi-sphere support vector data description for outliers detection
on multi-distribution data. In 2009 IEEE international conference
on data mining workshops, pages 82–87. IEEE, 2009.

[Xu et al., 2018] Haowen Xu, Wenxiao Chen, Nengwen Zhao,
Zeyan Li, Jiahao Bu, Zhihan Li, Ying Liu, Youjian Zhao, Dan
Pei, Yang Feng, et al. Unsupervised anomaly detection via variational
auto-encoder for seasonal KPIs in web applications. In
Proceedings of the International Conference on World Wide Web
(WWW), pages 187–196. International World Wide Web Conferences
Steering Committee, International World Wide Web Conferences
Steering Committee, 2018.

[Yamanishi et al., 2004] Kenji Yamanishi, Jun-ichi Takeuchi, Graham
Williams, and Peter Milne. On-Line Unsupervised Outlier
Detection Using Finite Mixtures with Discounting Learning Algorithms.
8(3):275–300, 2004.

[Yang and Liao, 2017] Chao-Lung Yang and Wei-Ju Liao. Adjacent
mean difference (amd) method for dynamic segmentation in
time series anomaly detection. In 2017 IEEE/SICE International
Symposium on System Integration (SII), pages 241–246. IEEE,
2017.

[Yankov et al., 2008] Dragomir Yankov, Eamonn Keogh, and
Umaa Rebbapragada. Disk aware discord discovery: Finding
unusual time series in terabyte sized datasets. Knowledge and
Information Systems, 17:241–262, 2008.

[Yao et al., 2010] Yuan Yao, Abhishek Sharma, Leana Golubchik,
and Ramesh Govindan. Online anomaly detection for sensor
systems: A simple and efficient approach. Perform. Eval.,
67(11):1059–1075, nov 2010.

[Yeh et al., 2016] Chin-Chia Michael Yeh, Yan Zhu, Liudmila
Ulanova, Nurjahan Begum, Yifei Ding, Hoang Anh Dau,
Diego Furtado Silva, Abdullah Mueen, and Eamonn Keogh. Matrix
profile i: all pairs similarity joins for time series: a unifying
view that includes motifs, discords and shapelets. In ICDM,
pages 1317–1322. IEEE, 2016.

[Yu et al., 2014] Yufeng Yu, Yuelong Zhu, Shijin Li, and DingshengWan.
Time Series Outlier Detection Based on SlidingWindow
Prediction. 2014:1–14, 2014.

[Zhang et al., 2007] Rui Zhang, Shaoyan Zhang, Sethuraman
Muthuraman, and Jianmin Jiang. One class support vector
machine for anomaly detection in the communication network
performance data. In Proceedings of the Conference
on Applied Electromagnetics, Wireless and Optical Communications
(ELECTROSCIENCE), ELECTROSCIENCE’07, pages
31–37. World Scientific and Engineering Academy and Society
(WSEAS), 2007.

[Zhang et al., 2019] Chuxu Zhang, Dongjin Song, Yuncong Chen,
Xinyang Feng, Cristian Lumezanu, Wei Cheng, Jingchao Ni,
Bo Zong, Haifeng Chen, and Nitesh V. Chawla. A Deep Neural
Network for Unsupervised Anomaly Detection and Diagnosis
in Multivariate Time Series Data. In AAAI, volume 33, pages
1409–1416, 2019.

[Zhang et al., 2020] Chunkai Zhang, Shaocong Li, Hongye Zhang,
and Yingyang Chen. VELC: A New Variational AutoEncoder
Based Model for Time Series Anomaly Detection. 2020.

[Zhao et al., 2020] Hang Zhao, Yujing Wang, Juanyong Duan,
Congrui Huang, Defu Cao, Yunhai Tong, Bixiong Xu, Jing Bai,
Jie Tong, and Qi Zhang. Multivariate time-series anomaly detection
via graph attention network. In ICDM, pages 841–850.
IEEE, 2020.

[Zhu et al., 2016] Yan Zhu, Zachary Zimmerman, Nader Shakibay
Senobari, Chin-Chia Michael Yeh, Gareth Funning, Abdullah
Mueen, Philip Brisk, and Eamonn Keogh. Matrix profile ii: Exploiting
a novel algorithm and gpus to break the one hundred million
barrier for time series motifs and joins. In 2016 IEEE 16th international
conference on data mining (ICDM), pages 739–748.
IEEE, 2016.

[Zhu et al., 2018] Yan Zhu, Chin-Chia Michael Yeh, Zachary Zimmerman,
Kaveh Kamgar, and Eamonn Keogh. Matrix profile xi:
Scrimp++: time series motif discovery at interactive speeds. In
2018 IEEE International Conference on Data Mining (ICDM),
pages 837–846. IEEE, 2018.

[Zimmerman et al., 2019a] Zachary Zimmerman, Kaveh Kamgar,
Nader Shakibay Senobari, Brian Crites, Gareth Funning, Philip
Brisk, and Eamonn Keogh. Matrix profile xiv: scaling time series motif discovery with gpus to break a quintillion pairwise comparisons
a day and beyond. In Proceedings of the ACM Symposium
on Cloud Computing, pages 74–86, 2019.

[Zimmerman et al., 2019b] Zachary Zimmerman, Nader Shakibay
Senobari, Gareth Funning, Evangelos Papalexakis, Samet Oymak,
Philip Brisk, and Eamonn Keogh. Matrix profile xviii: time
series mining in the face of fast moving streams using a learned
approximate matrix profile. In 2019 IEEE International Conference
on Data Mining (ICDM), pages 936–945. IEEE, 2019.