package com.example.mtcnn;

import static com.example.mtcnn.FaceUtils.assetFilePath;
import static com.example.mtcnn.FaceUtils.convert_to_square;
import static com.example.mtcnn.FaceUtils.crop_and_resize;
import static com.example.mtcnn.FaceUtils.getCls;
import static com.example.mtcnn.FaceUtils.getOffset;
import static com.example.mtcnn.FaceUtils.getPoint;
import static com.example.mtcnn.FaceUtils.minmum;
import static com.example.mtcnn.FaceUtils.nms;
import static com.example.mtcnn.FaceUtils.nonzero;
import static com.example.mtcnn.FaceUtils.transebox;
import static com.example.mtcnn.FaceUtils.zhengheonetbox;
import static com.example.mtcnn.FaceUtils.zhenghernetbox;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.Camera;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.content.ContextCompat;
import androidx.lifecycle.LifecycleOwner;

import android.annotation.SuppressLint;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.ImageFormat;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.media.Image;
import android.os.Bundle;
import android.util.DisplayMetrics;
import android.util.Log;
import android.util.Size;
import android.widget.ImageView;
import android.widget.TextView;

import com.google.common.util.concurrent.ListenableFuture;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.Vector;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

public class RealTime extends AppCompatActivity {
    private ListenableFuture<ProcessCameraProvider> cameraProviderFuture;
    Bitmap bitmap_vedio = null;
    DisplayMetrics displayMetrics = null;
    PreviewView previewView;
    float depth_score = 0;
    float[] all_depth_score = null;
    int[] all_is_live_face = null;
    TextView textView;
    float[] face_live_mean = new float[]{0.485f, 0.456f, 0.406f};
    float[] face_live_std = new float[]{0.229f, 0.224f, 0.225f};
    float[] face_mean = new float[]{0.5f, 0.5f, 0.5f};
    float[] face_std = new float[]{0.5f, 0.5f, 0.5f};
    List<Box> boxes = null;
    int is_live_face=0;
    FaceBoxView faceBoxView;
    Module module_live = null;
    Module p_net = null;
    Module r_net = null;
    Module o_net = null;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_real_time);
        previewView = (PreviewView) findViewById(R.id.realtimecamera);
        faceBoxView = (FaceBoxView) findViewById(R.id.resultView);
        textView =  (TextView) findViewById(R.id.textview);
        displayMetrics = new DisplayMetrics();
        getWindowManager().getDefaultDisplay().getMetrics(displayMetrics);
        try {
            //加载mtcnn人脸检测模型
            p_net = Module.load(assetFilePath(this, "p_net.pt"));
            r_net = Module.load(assetFilePath(this, "r_net.pt"));
            o_net = Module.load(assetFilePath(this, "o_net.pt"));
            //加载活体检测模型
            module_live = Module.load(assetFilePath(this,"p1_mobile.ptl"));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        cameraProviderFuture = ProcessCameraProvider.getInstance(this);
        cameraProviderFuture.addListener(()->{
            try{
                ProcessCameraProvider cameraProvider = cameraProviderFuture.get();
                startCamera(cameraProvider);
            } catch (ExecutionException | InterruptedException e){
            }
        }, ContextCompat.getMainExecutor(this));
    }
    Executor executor = Executors.newSingleThreadExecutor();
    void startCamera(@NonNull ProcessCameraProvider cameraProvider){
        Preview preview = new Preview.Builder().build();
        CameraSelector cameraSelector = new CameraSelector.Builder().requireLensFacing(CameraSelector.LENS_FACING_BACK).build();
        preview.setSurfaceProvider(previewView.getSurfaceProvider());
        ImageAnalysis imageAnalysis = new ImageAnalysis.Builder().setTargetResolution(new Size(displayMetrics.widthPixels,displayMetrics.heightPixels))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build();
        imageAnalysis.setAnalyzer(executor, new ImageAnalysis.Analyzer() {
            @Override
            public void analyze(@NonNull ImageProxy image) {
                int rotation = image.getImageInfo().getRotationDegrees();
                analyzeImage(image, rotation);
                image.close();
            }
        });
        Camera camera = cameraProvider.bindToLifecycle((LifecycleOwner) this, cameraSelector, preview, imageAnalysis);
    }
    void analyzeImage(ImageProxy image, int rotation){
        // 传入图片，传出分类结果，并显示在界面当中
        bitmap_vedio = getBitmap(image);
        long start= System.currentTimeMillis();
        boxes = detect(bitmap_vedio);
        long time = System.currentTimeMillis()-start;
        faceBoxView.setResults(boxes,bitmap_vedio,all_is_live_face,all_depth_score);
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                if(all_is_live_face!=null) {
                    all_is_live_face = null;
                    all_depth_score=null;
                    textView.setText("耗时 = " + String.valueOf(time) + "ms\nthreshold = 0.2783");
                }
                    else {
                    textView.setText("未检测到人脸\nthreshold = 0.2783");
                }
            }
        });

    }
    public static Bitmap getBitmap(ImageProxy image) {
        FrameMetadata frameMetadata =
                new FrameMetadata.Builder()
                        .setWidth(image.getWidth())
                        .setHeight(image.getHeight())
                        .setRotation(image.getImageInfo().getRotationDegrees())
                        .build();

        @SuppressLint("UnsafeOptInUsageError") ByteBuffer nv21Buffer =
                yuv420ThreePlanesToNV21(image.getImage().getPlanes(), image.getWidth(), image.getHeight());
        return getBitmap(nv21Buffer, frameMetadata);
    }
    public static Bitmap getBitmap(ByteBuffer data, FrameMetadata metadata) {
        data.rewind();
        byte[] imageInBuffer = new byte[data.limit()];
        data.get(imageInBuffer, 0, imageInBuffer.length);
        try {
            YuvImage image =
                    new YuvImage(
                            imageInBuffer, ImageFormat.NV21, metadata.getWidth(), metadata.getHeight(), null);
            ByteArrayOutputStream stream = new ByteArrayOutputStream();
            image.compressToJpeg(new Rect(0, 0, metadata.getWidth(), metadata.getHeight()), 80, stream);

            Bitmap bmp = BitmapFactory.decodeByteArray(stream.toByteArray(), 0, stream.size());

            stream.close();
            return rotateBitmap(bmp, metadata.getRotation(), false, false);
        } catch (Exception e) {
            Log.e("VisionProcessorBase", "Error: " + e.getMessage());
        }
        return null;
    }
    private static Bitmap rotateBitmap(
            Bitmap bitmap, int rotationDegrees, boolean flipX, boolean flipY) {
        Matrix matrix = new Matrix();
        // Rotate the image back to straight.
        matrix.postRotate(rotationDegrees);
        // Mirror the image along the X or Y axis.
        matrix.postScale(flipX ? -1.0f : 1.0f, flipY ? -1.0f : 1.0f);
        Bitmap rotatedBitmap =
                Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);
        // Recycle the old bitmap if it has changed.
        if (rotatedBitmap != bitmap) {
            bitmap.recycle();
        }
        return rotatedBitmap;
    }
    private static ByteBuffer yuv420ThreePlanesToNV21(
            Image.Plane[] yuv420888planes, int width, int height) {
        int imageSize = width * height;
        byte[] out = new byte[imageSize + 2 * (imageSize / 4)];

        if (areUVPlanesNV21(yuv420888planes, width, height)) {
            // Copy the Y values.
            yuv420888planes[0].getBuffer().get(out, 0, imageSize);
            ByteBuffer uBuffer = yuv420888planes[1].getBuffer();
            ByteBuffer vBuffer = yuv420888planes[2].getBuffer();
            // Get the first V value from the V buffer, since the U buffer does not contain it.
            vBuffer.get(out, imageSize, 1);
            // Copy the first U value and the remaining VU values from the U buffer.
            uBuffer.get(out, imageSize + 1, 2 * imageSize / 4 - 1);
        } else {
            // Fallback to copying the UV values one by one, which is slower but also works.
            // Unpack Y.
            unpackPlane(yuv420888planes[0], width, height, out, 0, 1);
            // Unpack U.
            unpackPlane(yuv420888planes[1], width, height, out, imageSize + 1, 2);
            // Unpack V.
            unpackPlane(yuv420888planes[2], width, height, out, imageSize, 2);
        }

        return ByteBuffer.wrap(out);
    }
    private static boolean areUVPlanesNV21(Image.Plane[] planes, int width, int height) {
        int imageSize = width * height;
        ByteBuffer uBuffer = planes[1].getBuffer();
        ByteBuffer vBuffer = planes[2].getBuffer();
        // Backup buffer properties.
        int vBufferPosition = vBuffer.position();
        int uBufferLimit = uBuffer.limit();
        // Advance the V buffer by 1 byte, since the U buffer will not contain the first V value.
        vBuffer.position(vBufferPosition + 1);
        // Chop off the last byte of the U buffer, since the V buffer will not contain the last U value.
        uBuffer.limit(uBufferLimit - 1);
        // Check that the buffers are equal and have the expected number of elements.
        boolean areNV21 =
                (vBuffer.remaining() == (2 * imageSize / 4 - 2)) && (vBuffer.compareTo(uBuffer) == 0);
        // Restore buffers to their initial state.
        vBuffer.position(vBufferPosition);
        uBuffer.limit(uBufferLimit);
        return areNV21;
    }
    private static void unpackPlane(
            Image.Plane plane, int width, int height, byte[] out, int offset, int pixelStride) {
        ByteBuffer buffer = plane.getBuffer();
        buffer.rewind();

        // Compute the size of the current plane.
        // We assume that it has the aspect ratio as the original image.
        int numRow = (buffer.limit() + plane.getRowStride() - 1) / plane.getRowStride();
        if (numRow == 0) {
            return;
        }
        int scaleFactor = height / numRow;
        int numCol = width / scaleFactor;
        // Extract the data in the output buffer.
        int outputPos = offset;
        int rowStart = 0;
        for (int row = 0; row < numRow; row++) {
            int inputPos = rowStart;
            for (int col = 0; col < numCol; col++) {
                out[outputPos] = buffer.get(inputPos);
                outputPos += pixelStride;
                inputPos += plane.getPixelStride();
            }
            rowStart += plane.getRowStride();
        }
    }

    public  List<Box> detect(Bitmap bitmap){
        Bitmap first = bitmap.copy(Bitmap.Config.ARGB_8888, true);;
        int minside = minmum(bitmap.getHeight(),bitmap.getWidth());
        List<Box> p_net_box = new ArrayList<>();
        List<Box> boxes1 = new ArrayList<>();
        //获得大于0.6的索引
        List<Integer> index = new ArrayList<>();
        //        根据索引获得置信度，注意清空
        List<Float> _facecls = new ArrayList<>();
        //        根据索引获得坐标偏移，注意清空
        List<Offset> _facebox = new ArrayList<>();
        int w=0,h=0;
        float scale = 1.0f;
        float min_face_size = 400f;
        while (scale>(12.0f/min_face_size)){
            scale*=0.709f;
            w = bitmap.getWidth();
            h = bitmap.getHeight();
            //获取想要缩放的matrix
            Matrix matrix = new Matrix();
            matrix.postScale(0.709f, 0.709f);
            //获取新的bitmap
            bitmap = Bitmap.createBitmap(bitmap, 0, 0, w, h, matrix, true);
            minside = minmum(bitmap.getHeight(),bitmap.getWidth());
        }
        while(minside>12) {
            final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(bitmap,
                    face_mean, face_std);
            final IValue[] outputTensor = p_net.forward(IValue.from(inputTensor)).toTuple();
            IValue bbox = outputTensor[0];
            Tensor boxt = bbox.toTensor();
            List<String> sp = new ArrayList<>();
            for (int i = 0; i < boxt.toString().split(", ").length; i++) {
                sp.add(boxt.toString().split(", ")[i]);
            }
            int width = Integer.parseInt(sp.get(3).split("]")[0]);
            float[] facecls = boxt.getDataAsFloatArray();
            IValue cls = outputTensor[1];
            Tensor clst = cls.toTensor();
            float[] facebox = clst.getDataAsFloatArray();
            index.addAll(nonzero(facecls, 0.6f));
            _facecls.addAll(getCls(index, facecls));
            _facebox.addAll(getOffset(index, facecls.length, facebox));
            //        将置信度与偏移整合变为[x1,y1,x2,y2,cls]格式
            boxes1.addAll(transebox(width, index, _facecls, _facebox, scale, 2, 12));
            scale*=0.709f;
            w = bitmap.getWidth();
            h = bitmap.getHeight();
            //获取想要缩放的matrix
            Matrix matrix = new Matrix();
            matrix.postScale(0.709f, 0.709f);
            //获取新的bitmap
            bitmap = Bitmap.createBitmap(bitmap, 0, 0, w, h, matrix, true);
            minside = minmum(bitmap.getHeight(),bitmap.getWidth());
            index.clear();
            _facecls.clear();
            _facebox.clear();

        }

        p_net_box.addAll(nms(boxes1, 0.6f, false));
//p_net网络结束
//        r_net开始
        List<Box> p_r_box = new ArrayList<>();
        w = first.getWidth();
        h = first.getHeight();
        p_r_box.addAll(convert_to_square(p_net_box,w,h));
        for(int i =0;i<p_r_box.size();i++){
            if(p_r_box.get(i).getX1()<0){
                Log.e("dasdasdasdas",p_r_box.get(i).getX1()+"");
            }
        }
        Bitmap bt=null;
        List<Box> p_index_r = new ArrayList<>();
        for(int i=0;i<p_r_box.size();i++) {
            bt = crop_and_resize(first,p_r_box.get(i),24 );
            final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(bt,
                    face_mean, face_std);
            final IValue[] outputTensor = r_net.forward(IValue.from(inputTensor)).toTuple();
            IValue bbox = outputTensor[0];
            Tensor boxt = bbox.toTensor();
            float[] facecls = boxt.getDataAsFloatArray();
            IValue cls = outputTensor[1];
            Tensor clst = cls.toTensor();
            float[] facebox = clst.getDataAsFloatArray();
            //注意清空
            index.addAll(nonzero(facecls, 0.7f));
            //        根据索引获得置信度
            if (index.size()>0){
                p_index_r.add(p_r_box.get(i));
            }
            _facecls.addAll(getCls(index, facecls));
            //        根据索引获得坐标偏移
            _facebox.addAll(getOffset(index, 1,facebox));
            index.clear();
        }
        //整合
        List<Box> boxes =new ArrayList<>();
        List<Box> r_net_box = new ArrayList<>();
        boxes.addAll(zhenghernetbox(p_index_r,_facecls,_facebox));
        r_net_box.addAll(nms(boxes,0.7f,false));
        _facecls.clear();
        _facebox.clear();
//r_net结束
        List<Point> _facepointo = new ArrayList<>();
        List<Box> r_o_box = new ArrayList<>();
        r_o_box.addAll(convert_to_square(r_net_box,w,h));
        Bitmap bt1=null;
        List<Box> r_index_o = new ArrayList<>();
        for(int i=0;i<r_o_box.size();i++) {
            bt1 = crop_and_resize(first,r_o_box.get(i),48 );
            final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(bt1,
                    face_mean, face_std);
            final IValue[] outputTensor = o_net.forward(IValue.from(inputTensor)).toTuple();
            IValue bbox = outputTensor[0];
            Tensor boxt = bbox.toTensor();
            float[] facecls = boxt.getDataAsFloatArray();
            IValue cls = outputTensor[1];
            Tensor clst = cls.toTensor();
            float[] facebox = clst.getDataAsFloatArray();
            IValue poin = outputTensor[2];
            Tensor point = poin.toTensor();
            float [] points = point.getDataAsFloatArray();
            //注意清空
            index.addAll(nonzero(facecls, 0.95f));
//            //        根据索引获得置信度
            if (index.size()>0){
                r_index_o.add(r_o_box.get(i));
            }
            _facecls.addAll(getCls(index, facecls));
            //        根据索引获得坐标偏移
            _facepointo.addAll(getPoint(index,points));
            _facebox.addAll(getOffset(index, 1,facebox));
            index.clear();
        }
        List<Box> rbox = new ArrayList<>();
        rbox.addAll(zhengheonetbox(r_index_o,_facecls,_facebox,_facepointo));
        List<Box> o_net_box = new ArrayList<>();
        o_net_box.addAll(nms(rbox,0.95f,true));
        //o_net结束
        if(o_net_box.size()==0){
            is_live_face=0;
        }
        else {
            all_depth_score = new float[o_net_box.size()];
            all_is_live_face = new int[o_net_box.size()];
            System.out.println(o_net_box.size());
            for (int i=0;i<o_net_box.size();i++) {
                //对图像进行人脸裁剪
                Bitmap live_bitmap = Bitmap.createBitmap(first, Math.max(o_net_box.get(i).getX1(), 0), Math.max(o_net_box.get(i).getY1(), 0), Math.min(o_net_box.get(i).getX2() - o_net_box.get(i).getX1(), first.getWidth() - Math.max(o_net_box.get(i).getX1(), 0)), Math.min(o_net_box.get(i).getY2() - o_net_box.get(i).getY1(), first.getHeight() - Math.max(o_net_box.get(i).getY1(), 0)));
                //调整尺寸为224*224
                live_bitmap = Bitmap.createScaledBitmap(live_bitmap, 224, 224, true);
                //送入人脸活体检测模型进行金策
                all_is_live_face[i] = is_live(live_bitmap);
                all_depth_score[i] = depth_score;
            }

        }
        return o_net_box;
    }
    public int is_live(Bitmap bitmap){
        final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(bitmap,
                face_live_mean,face_live_std);
        final Tensor outputTensor = module_live.forward(IValue.from(inputTensor)).toTensor();
        float[] scores = outputTensor.getDataAsFloatArray();
        depth_score = get_depth_scores(scores);
        if(depth_score>0.2783f){
            return 1;
        }
        else {
            return 2;
        }
    }
    public static float get_depth_scores(float score[]){
        float depth_scores = 0.0f;
        for(int i = 0;i<score.length;i++){
            depth_scores +=score[i];
        }
        return depth_scores/score.length;
    }
}