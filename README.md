<h1> Pure Python bitmap decoder/encoder</h1>

This is just a little bitmap encoder/decoder I wrote to get a better understanding of the bitmap file format.

This is pretty inefficient but gets the job done. It can read and write bitmap files.

Current functionality:
-  Read images
- Write images
- Convert to grayscale
- Center crop

<h3>Examples</h3>

```
import bmp
bitmap = bmp.BitmapReader('frame.bmp')
bitmap.grayscale()
bitmap.write('frame_gray.bmp')
```

