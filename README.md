# DIP
## DEMOSAIC Image Processing module

split channel and demix  
    
![befor process](/img/TEST_MIXED_uint8.png) to ![after process](/img/TEST_DEMIXED_uint8.png)
  
  
  to run,
  
  ```
  python process.py --src_path C:/.../raw_image.tiff --ratio_path C:/.../ratio.csv --dark_path(optional) C:/.../dark.tiff 
  --new_name newfilename.tiff --tune(optional)
  ```

  to use class Image40,
  ```
  from utils.DIP import Image40
  ```
## 07.06
  updated STA analysis tool(jupyter notebook pipleine)
