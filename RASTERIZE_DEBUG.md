# Rasterization Debug Notes — 0 Pixels Issue (Historical)

> **Status:** *Historical debugging notes (Hebrew).* Kept as a checklist for
> diagnosing CRS / transform mismatches when vector overlays produce zero
> output pixels. The current code in
> [backend/app/core.py:rasterize_vectors_onto_classification](backend/app/core.py)
> calls `gdf.to_crs(raster_crs)` (not just `set_crs`) before rasterising, so
> this should rarely fire — but the Hebrew checklist below remains useful when
> the bug recurs from upstream data with a malformed CRS.



## הבעיה
- ✓ 26,997 גיאומטריות חוקיות
- ✗ 0 pixels נעשו rasterize

## סיבות אפשריות

### 1. **CRS / Projection Mismatch** (הסיבה הסבירה ביותר)
הגיאומטריות והרסטר נמצאים בשיטות קואורדינטות שונות:
- את תראה בלוג: `LOCAL_CS["WGS 84 / Pseudo-Mercator"...`
- זה אומר שהקואורדינטות בשיטה מקומית
- גיאומטריות עשויות להיות בלוורט עם קואורדינטות בעמודות/שורות
- רסטר עשוי להיות בMercator או UTM

### 2. **Transform לא נכון**
ה-transform (affine matrix) עשוי להיות לא תואם לרסטר בפועל

### 3. **גבולות לא חופפים**
הגיאומטריות בחוץ מהגבולות הפיזיים של הרסטר

## איך לאבחן

1. **הרץ את script זה:**
```bash
python check_crs.py
```

זה יראה:
- ✓/✗ האם CRS תואם בדיוק
- ✓/✗ האם אפשר להשתנות מ-CRS של וקטור ל-CRS של רסטר
- הגבולות של שניהם

2. **אם CRS לא מתאים:**
   - עדכן את thecalssification_path להיות עם CRS נכון
   - או change את `to_crs()` להתחיל

## פתרונות אפשריים

1. **אם בעיית CRS:**
```python
# במקום:
gdf.set_crs(crs, allow_override=True)

# קרא:
gdf = gdf.to_crs(crs)
```

2. **אם בעיית Transform:**
   - בדוק שה-Transform מתאים לגיאומטריות

3. **אם Projection לוקלית:**
   - אולי צריך להשתמש בפיקסלים ישירות בלי projection
