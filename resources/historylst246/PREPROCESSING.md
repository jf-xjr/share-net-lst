# Fixed observation and scoring definitions

The distributed arrays, source identities and SHA-256 values define version 1.
No source reselection or reprocessing occurs when a method is trained. These notes
describe their construction; this package is not a raw-scene acquisition service.
Original preprocessing source hashes and fitted normalization values accompany
the release in `provenance/`.

**Grids and reference.** Each crop is 640 × 640 delivered Landsat pixels at 30 m,
aggregated to 160 × 160 at 120 m. `transform120` is the six/nine-coefficient affine
transform in the supplied CRS, in rasterio/Affine order; the upper-left corner
and orientation are preserved. The 480 m grid groups 4 × 4 of the 120 m cells.
Landsat's delivered 30 m LST pixels do not constitute independent 30 m thermal
measurements. The target is a Landsat Collection 2 Level 2 temperature product.

**Query support and coarse generation.** Query QA rejects cloud/fill/shadow/snow
bits 0–5, followed by one delivered-pixel dilation, radiometric saturation,
invalid reflectance or temperature, and the registered aerosol states. Accepted
temperature is 240–380 K with reported ST_QA at most 3 K (the current query implementation has no separate ST_QA > 0 test). The temperature
conversion is `DN * 0.00341802 + 149` K. A 120 m target requires at least 12 accepted
30 m pixels; an observed 480 m parent requires 12 supported 120 m cells. Coarse
temperature is the mean over the supported 120 m children, rather than a weighted
mean over all delivered 30 m children. Interpolation uses the coarse field and its
availability weights, with the registered 0.5 interpolation-weight threshold.
The supplied `support` retains this original thermal/QA-dependent construction.
The supplied `formal` additionally applies the original valid and urban-eligibility
rules and is the only primary scoring domain. It must not be reconstructed using
a new threshold from current predictions.

**Current predictors.** The original thermally screened reflectance representation
was replaced upstream by exact-query optical-only sidecars. Six Landsat SR bands
(blue, green, red, NIR, SWIR1, SWIR2) use their own optical QA and the conversion
`DN * 0.0000275 - 0.2`. The 52 channels, named in the manifest, are: coarse-derived
temperature, interpolation weight, six Fit603-normalized bands, six scene
median/IQR band anomalies clipped to [-8,8], NDVI/NDBI/MNDWI/BSI, optical validity,
built and water fractions and land-cover coverage, plus 30 Fit603-normalized
optical texture statistics. Land cover comes from the 2021
`io-lulc-annual-v02` product. These fractional cells have been reprojected and
aggregated from source classes. Texture includes band standard deviation and
quartiles, and mean/standard deviation/quartiles for three indices.

The context vector includes season, Landsat platform, solar geometry and NASA
POWER UTC daily meteorology. Its 15 channel names are in the manifest. Seven daily
weather variables are T2M_MAX, T2M_MIN, RH2M, WS2M, ALLSKY_SFC_SW_DWN, PRECTOTCORR
and GWETTOP. Maximum temperature appears in the legacy context and again in the
seven-variable weather extension, using their preserved normalization values.
This is a retrospective task; complete daily weather is not an overpass-time
forecast. Latitude/longitude columns are absent from the supplied vector.

**Current emissivity.** E/EMSD use a joint mask: `6000 <= E_DN <= 10000` and
`0 < EMSD_DN <= 10000`, both scaled by 0.0001. The mean, standard deviation, mean
EMSD and count are computed on that mask in each 4 × 4 delivered block. Continuous
missing values share the nearest valid 120 m donor; availability stays zero.
The four channels are `(mean(E)-.98)/.01`, `sd(E)/.01`,
`log1p(mean(EMSD)/.01)` and `count/16`. This availability is independent of thermal
availability; the temperature band is not read to construct these predictors.

**Past source selection.** Slots 0–2 each select an older 2018/2019/2020
May–September Landsat 8 acquisition with catalog cloud cover at most 60%, ordered
by cloud cover, acquisition time and item identity. Slots 3–5 select from the same
years and season with cloud cover at most 40%, ranked by circular calendar distance
to the query, then cloud/time/identity. A repeated annual/seasonal acquisition
becomes an empty slot. Slots 6–8 select distinct Landsat 8/9 overpasses 8–64 days
before the query, catalog cloud cover at most 40%, full crop and required assets
available, ranked by age/cloud/time/identity. Campaign query identities and
equivalent aliases were excluded. Exact selected identities and UTCs, including
unavailable/empty sources, are retained per query; reranking is not needed.

**Historical encoding.** Raw source rasters are reprojected by nearest neighbor to
the fixed 640 × 640 grid. Historical clear support requires `ST_DN>0`, temperature
240–380 K, `0<ST_QA_DN<=300`, zero QA_RADSAT, and rejection of QA_PIXEL bits 0–5 with
one-pixel dilation. A historical 120 m cell may contain only one clear delivered
pixel; `count/16` exposes this partial coverage. Means of T and ST_QA use only
clear source pixels. Missing means share the nearest nonempty historical cell.
The parent-anomaly predictor subtracts the mean of the filled 4 × 4 parent.
Historical emissivity is encoded with its separate joint E/EMSD availability.

The nine channels are `(filled_T-300)/20`, `filled_parent_anomaly/5`, thermal
`count/16`, `ST_QA_K/3`, `(filled_E-.98)/.01`, emissivity `count/16`, sine and cosine
of acquisition DOY with period 365.25, and positive source age/3652.5 days. A wholly
thermal-empty source is all zero, including time. The decoded temperature is the
stored float32 normalized value transformed back to K, not a bit-exact recovery
of the original DN mean. The observed decoder discards nearest fill outside actual
source coverage. Native 30 m QA arrangements and discarded E-only values of a
wholly thermal-empty slot are not recoverable from this version.

**Common output operator.** For each observed parent, add the difference between
its supplied coarse temperature and the mean prediction on its actual supported
children to each supported child. An unobserved parent receives no correction.
The NumPy implementation uses FP64; unsupported cells are NaN. This preserves the
observed mean and changes no within-parent contrast. It ensures arithmetic
consistency with this controlled input, not radiance conservation across sensors.

**Evaluation.** `target`, `valid`, and `formal` are separated from inputs. A method
must output K on every formal pixel and preserve query order. Compute per-scene
RMSE, MAE, bias and MSE; average within city, within each of China/Europe/US, and
then across the three regions. Report whether projection was applied. The supplied
dates are three samples per city, not a dense daily sequence. City-isolated splits
do not imply that every source granule is independent: some nonoverlapping crops
share an earlier global Landsat acquisition. These distinctions matter when
interpreting cross-city scores or estimating uncertainty.
