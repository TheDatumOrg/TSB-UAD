# First, we need data

Multiple labeled time series datasets exist for anomaly detection. We provide below a detailed list of these datasets, which are included in TSB-UAD benchmarks and can be downloaded [here](https://www.thedatum.org/datasets/TSB-UAD-Public.zip).

## Datasets Descriptions

| Dataset    | Description|
|:--|:---------:|
|Dodgers| is a loop sensor data for the Glendale on-ramp for the 101 North freeway in Los Angeles and the anomalies represent unusual traffic after a Dodgers game.|
|ECG| is a standard electrocardiogram dataset and the anomalies represent ventricular premature contractions. We split one long series (MBA_ECG14046) with length ∼ 1e7) to 47 series by first identifying the periodicity of the signal.|
|IOPS| is a dataset with performance indicators that reflect the scale, quality of web services, and health status of a machine.|
|KDD21| is a composite dataset released in a recent SIGKDD 2021 competition with 250 time series.|
|MGAB| is composed of Mackey-Glass time series with non-trivial anomalies. Mackey-Glass time series exhibit chaotic behavior that is difficult for the human eye to distinguish.|
|NAB| is composed of labeled real-world and artificial time series including AWS server metrics, online advertisement clicking rates, real time traffic data, and a collection of Twitter mentions of large publicly-traded companies.|
|NASA-SMAP and NASA-MSL| are two real spacecraft telemetry data with anomalies from Soil Moisture Active Passive (SMAP) satellite and Curiosity Rover on Mars (MSL). We only keep the first data dimension that presents the continuous data, and we omit the remaining dimensions with binary data.|
|SensorScope| is a collection of environmental data, such as temperature, humidity, and solar radiation, collected from a typical tiered sensor measurement system.|
|YAHOO| is a dataset published by Yahoo labs consisting of real and synthetic time series based on the real production traffic to some of the Yahoo production systems.|
|Daphnet| contains the annotated readings of 3 acceleration sensors at the hip and leg of Parkinson’s disease patients that experience freezing of gait (FoG) during walking tasks.|
|GHL| is a Gasoil Heating Loop Dataset and contains the status of 3 reservoirs such as the temperature and level. Anomalies indicate changes in max temperature or pump frequency.|
|Genesis| is a portable pick-and-place demonstrator which uses an air tank to supply all the gripping and storage units.|
|MITDB| contains 48 half-hour excerpts of two-channel ambulatory ECG recordings, obtained from 47 subjects studied by the BIH Arrhythmia Laboratory between 1975 and 1979.|
|OPPORTUNITY (OPP)| is a dataset devised to benchmark human activity recognition algorithms (e.g., classiffication, automatic data segmentation, sensor fusion, and feature extraction). The dataset comprises the readings of motion sensors recorded while users executed typical daily activities.|
|Occupancy| contains experimental data used for binary classiffication (room occupancy) from temperature, humidity, light, and CO2. Ground-truth occupancy was obtained from time stamped pictures that were taken every minute.|
|SMD (Server Machine Dataset)| is a 5-week-long dataset collected from a large Internet company. This dataset contains 3 groups of entities from 28 different machines.|
|SVDB| includes 78 half-hour ECG recordings chosen to supplement the examples of  supraventricular arrhythmias in the MIT-BIH Arrhythmia Database.|


## Datasets characteristics

The following table summarizes different characteristics of the datasets.

| Dataset    | Count |	Avg length |	Avg number of anomalies |	Avg number of abnormal points |
|:--|:---------:|:-------:|:--------:|:-------:|
|Dodger     | 1     |	50400.0        |	133.0                       |	5612                              |
|ECG        | 53    |	230351.9       |	195.6                       |	15634                             |
|IOPS       | 58    |	102119.2       |	46.5                        |	2312.3                            |
|KDD21      | 250   |	77415.06       |	1.0                         |	196.5                             |
|MGAB       | 10    |	100000.0       |	10.0                        |	200.0                             |
|NAB        | 58    |	6301.7         |	2.0                         |	575.5                             |
|SensorScope| 23    |	27038.4        |	11.2                        |	6110.4                            |
|YAHOO      | 367   |	1561.2         |	5.9                         |	10.7                              |
|NASA-MSL   | 27    |	2730.7         |	1.33                        |	286.3                             |
|NASA-SMAP  | 54    |	8066.0         |	1.26                        |	1032.4                            |
|Daphnet    | 45    |	21760.0        |	7.6                         |	2841.0                            |
|GHL        | 126   |	200001.0       |	1.2                         |	388.8                             |
|Genesis    | 6     |	16220.0        |	3.0                         |	50.0                              |
|MITDB      | 32    |	650000.0       |	210.1                       |	72334.3                           |
|OPP        | 465   |	31616.9        |	2.0                         |	1267.3                            |
|Occupancy  | 10    |	5725.8         |	18.3                        |	1414.5                            |
|SMD        | 281   |	25562.3        |	10.4                        |	900.2                             |
|SVDB       | 115   |	230400.0       |	208.0                       |	27144.5                           |

You may find more details (and the references) in this [paper](https://www.paparrizos.org/papers/PaparrizosVLDB22a.pdf).