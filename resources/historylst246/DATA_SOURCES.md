# Source attribution and terms

This derived-data release accompanies the public
[SHaRe-Net repository](https://github.com/jf-xjr/share-net-lst). No DOI has been
assigned. The processed arrays remain in a private Google Drive folder as
described in the repository download guide. Source attribution travels with
redistributed derivatives.
The arrays contain values and public scene identities, not access credentials.

| Component | Source and modifications | Provider terms |
|---|---|---|
| Query and historical temperatures, reflectance, quality and emissivity | USGS Landsat Collection 2 Level 2, obtained through Microsoft Planetary Computer; cropped, reprojected, masked, aggregated and encoded as described in PREPROCESSING.md | USGS states there are no restrictions on Landsat use or redistribution; retain credit to USGS. [USGS policy](https://www.usgs.gov/faqs/are-there-any-restrictions-use-or-redistribution-landsat-data) |
| Built/water fractions and coverage | 2021 annual 10 m land cover, `io-lulc-annual-v02`, produced by Impact Observatory, Microsoft and Esri; categorical data converted to crop fractions | CC BY 4.0; credit these producers and state the changes. [Provider collection and license](https://planetarycomputer.microsoft.com/api/stac/v1/collections/io-lulc-annual-v02), [license](https://creativecommons.org/licenses/by/4.0/) |
| Meteorological context | NASA POWER daily point service, UTC, AG community; selected dates and seven variables, normalized on Fit603 | Public NASA POWER data; acknowledge the POWER Project at NASA Langley Research Center. [Data service](https://power.larc.nasa.gov/docs/faqs/data/), [NASA open-data policy](https://www.earthdata.nasa.gov/engage/open-data-services-and-software/data-and-information-policy) |
| U-TAE source | Official VSainteuf/utae-paps, pinned commit in UPSTREAM.json; task adaptations listed there | MIT, original license retained in historylst/third_party/utae/LICENSE |
| ubESTARFM source | Official yuyi13/ubESTARFM, pinned commit and unchanged numerical source in fusion_reference/; existing past-only adapter | MIT, copyright Yi Yu, original license retained in fusion_reference/LICENSE |

Historical and query item IDs, acquisition UTCs and crop grids are in manifest.json.
Collection-level provider metadata and normalization are retained in provenance/.
An ASTER or ECOSTRESS reference product is not part of this package; neither is
the original WGAST source or its dataset. No third-party license has been inferred
from the mere presence of public code.

The new portable loader, rules, scorer and adaptation code may be used and modified
under the accompanying MIT license. This software license does not replace the
provider terms for the data or grant rights over independently sourced material.
