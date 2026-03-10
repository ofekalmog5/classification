declare module "georaster" {
  interface GeoRasterOptions {
    projection?: number;
    noDataValue?: number;
    xmin: number;
    xmax: number;
    ymin: number;
    ymax: number;
    pixelWidth: number;
    pixelHeight: number;
    width: number;
    height: number;
    numberOfRasters: number;
    values?: number[][][];
  }

  function parseGeoraster(
    input: ArrayBuffer | string | File | Blob,
    metadata?: Partial<GeoRasterOptions>
  ): Promise<GeoRasterOptions>;

  export default parseGeoraster;
}

declare module "georaster-layer-for-leaflet" {
  import * as L from "leaflet";

  interface GeoRasterLayerOptions extends L.GridLayerOptions {
    georaster?: any;
    georasters?: any[];
    resolution?: number;
    opacity?: number;
    debugLevel?: number;
    pixelValuesToColorFn?: (values: number[]) => string | null;
    customDrawFunction?: (context: {
      context: CanvasRenderingContext2D;
      values: number[][];
      x: number;
      y: number;
      width: number;
      height: number;
    }) => void;
  }

  class GeoRasterLayer extends L.GridLayer {
    constructor(options: GeoRasterLayerOptions);
    getBounds(): L.LatLngBounds;
    setOpacity(opacity: number): this;
  }

  export default GeoRasterLayer;
}
