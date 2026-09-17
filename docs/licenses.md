# License status and source attribution

Checked on 2026-09-17. This page records the licenses found for the materials
in this repository. It does not assign a new license to the repository,
processed dataset, or model checkpoints. Public access alone is not a general
permission to reuse or redistribute material; see
[GitHub's licensing guidance](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository).

## Project code and checkpoints

The portable task's software under `resources/historylst246/` has an
[MIT license](../resources/historylst246/LICENSE). Its third-party components
and source data retain their own terms. This scoped software license does not
establish a license for every file in the assembled repository.

No repository-wide license has been selected for the project's other code,
including the SHaRe-Net research implementations and release scripts. No
separate license declaration was found for the 17 checkpoint files listed in
the `selected-checkpoints` group of [release-assets.json](../release-assets.json).
That includes the three final SHaRe-Net models, their parents, the U-TAE
models, the MoCoLSK and THSTNet comparators, and the portable task's reference
models. The software licenses of their architectures should not be treated
as an explicit license grant for these checkpoint files. Their intended
reuse and redistribution terms remain to be specified by the rights holders.

## Third-party code with retained licenses

The four license files below were checked byte for byte against their
recorded upstream revisions. Original source-level notices remain relevant.

| Component and local location | Upstream revision and license |
|---|---|
| NAFNet and included BasicSR material: `research/sub04_20260911/naf_history/upstream/` | [NAFNet `2b4af71`](https://github.com/megvii-research/NAFNet/blob/2b4af71ebe098a92a75910c233a3965a3e93ede4/LICENSE): MIT for NAFNet, with the included BasicSR Apache-2.0 text; [retained license](../research/sub04_20260911/naf_history/upstream/LICENSE). |
| U-TAE: `resources/historylst246/historylst/third_party/utae/` | [utae-paps `987874e`](https://github.com/VSainteuf/utae-paps/blob/987874e27a98bb399277f2d55eb010744d26f881/LICENSE): MIT; [retained license](../resources/historylst246/historylst/third_party/utae/LICENSE). |
| ubESTARFM: `resources/historylst246/fusion_reference/` | [ubESTARFM `171cfbd`](https://github.com/yuyi13/ubESTARFM/blob/171cfbdc92a7e63ef8769face326f00cb650061a/LICENSE.md): MIT; [retained license](../resources/historylst246/fusion_reference/LICENSE). The upstream full text is `LICENSE.md`, distinct from its short R-package `LICENSE` metadata file. |
| timm: `research/near_neighbor_attribution_20260914/vendor_deps/timm/` | [timm `v0.4.12`](https://github.com/huggingface/pytorch-image-models/blob/v0.4.12/LICENSE): Apache-2.0 with individual source notices; [retained license](../research/near_neighbor_attribution_20260914/vendor_deps/timm-0.4.12.dist-info/LICENSE). For example, `models/swin_transformer.py` also retains Microsoft's MIT notice. |

This comparison verifies the retained license texts; it is not a claim that
every nested upstream dependency has been independently cleared. In
particular, a license for Microsoft's Swin Transformer does not license the
whole THSTNet implementation.

## Comparator permissions that remain unresolved

The following code is still present. Its complete redistribution permission
has **not** been confirmed by this audit.

- **THSTNet** — `research/near_neighbor_attribution_20260914/vendor/thstnet/`.
  The [official revision `c930199`](https://github.com/HuPengHua2021/THSTNet/tree/c930199305d81e546d637e3ffe3d40db995e5f4e)
  has no license file or license statement in its README. A citation request
  and a public GitHub repository are not a substitute for a license grant.
  Permission for redistributing this vendored implementation remains to be
  confirmed with its rights holders.
- **MoCoLSK via PGDM** — the three files in `baselines/PGDM/lib/moco/`
  exactly match [PGDM revision `956f76b`](https://github.com/cas222huan/PGDM/tree/956f76bbf565a78d3cd086abf479cb4609b9ac5c/lib/moco).
  PGDM has no license file or license statement in its README at that
  revision. Two files differ from the corresponding GrokLST implementation,
  so the upstream license cannot simply be assumed to cover the complete
  PGDM adaptations.
- **GrokLST and DynamicMLP provenance** — at [GrokLST revision `3ff71a2`](https://github.com/GrokCV/GrokLST/tree/3ff71a267895e6b1c584bf99e0da8d53a57e5b36),
  [LICENSE](https://github.com/GrokCV/GrokLST/blob/3ff71a267895e6b1c584bf99e0da8d53a57e5b36/LICENSE)
  contains Apache-2.0, while the
  [README license section](https://github.com/GrokCV/GrokLST/blob/3ff71a267895e6b1c584bf99e0da8d53a57e5b36/README.md#license)
  says CC BY-NC 4.0. This conflict has not been resolved. The copied
  `dynamic_mlp.py` also identifies
  [DynamicMLP](https://github.com/ylingfeng/DynamicMLP/tree/773197c5a0aaf15a4d4c8b54a4765ea41f784d3f)
  as a source; that upstream revision has no license file or README grant.
  Neither unrestricted commercial permission nor a complete permission
  chain is established for this comparator by the existing materials.

These findings concern the comparator code and the unresolved checkpoint
terms. They do not establish that the processed Landsat-derived data or
the independently implemented SHaRe-Net method is subject to those
comparator licenses. The baseline permissions need confirmation before
representing the complete repository as licensed for redistribution.

## Processed data

The release contains derivatives of three documented data sources, rather
than the datasets distributed by THSTNet, PGDM, or GrokLST. The transformations
and provider attribution are described in
[DATA_SOURCES.md](../resources/historylst246/DATA_SOURCES.md) and
[PREPROCESSING.md](../resources/historylst246/PREPROCESSING.md).

| Source | Verified provider terms and attribution |
|---|---|
| USGS Landsat Collection 2 Level 2 | USGS permits use and redistribution and requests source credit. Retain attribution to USGS. [USGS policy](https://www.usgs.gov/faqs/are-there-any-restrictions-use-or-redistribution-landsat-data). |
| Impact Observatory, Microsoft and Esri annual land cover (`io-lulc-annual-v02`) | The provider's collection metadata specifies CC BY 4.0. Retain producer credit, the license link, and the description of conversion to cropped fractions. [Provider metadata](https://planetarycomputer.microsoft.com/api/stac/v1/collections/io-lulc-annual-v02), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). |
| NASA POWER meteorological context | NASA's POWER support statement identifies the data as public domain. Credit NASA Langley Research Center's POWER Project and retain service/version/access information where recorded. The project requests redistribution notification; the guide does not state that prior approval is required. [NASA support statement](https://forum.earthdata.nasa.gov/viewtopic.php?t=82), [current referencing guide](https://power.larc.nasa.gov/docs/referencing/). |

The provider terms support public distribution of these processed data with
the stated attribution. They remain applicable when files are downloaded
or redistributed. This page does not replace them with the portable
software's MIT license or select a new license for the dataset compilation.
