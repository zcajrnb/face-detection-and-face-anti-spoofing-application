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

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class MainActivity extends AppCompatActivity implements View.OnClickListener {
    Button button_photo,button_vedio;
    float depth_score = 0f;
    ImageView imageView;
    TextView textView;
    Bitmap bitmap=null;
    Module module_live = null;
    Module p_net = null;
    Module r_net = null;
    Module o_net = null;
    int is_live_face=0;

    float[] face_live_mean = new float[]{0.485f, 0.456f, 0.406f};
    float[] face_live_std = new float[]{0.229f, 0.224f, 0.225f};
    float[] face_mean = new float[]{0.5f, 0.5f, 0.5f};
    float[] face_std = new float[]{0.5f, 0.5f, 0.5f};
    private final String[] REQUIRED_PERMISSIONS = new String[]{"android.permission.CAMERA"};
    private final String[] REQUIREAD_PERMISSIONS = new String[]{"android.permission.READ_EXTERNAL_STORAGE"};
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        if (!checkPermissionRead()) {
            grantPermissionRead(this);
        }
        if (!checkPermissionCamera()) {
            grantPermissionCamera(this);
        }
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

        button_photo = (Button) findViewById(R.id.photo);
        button_vedio = (Button) findViewById(R.id.vedio);
        imageView = (ImageView) findViewById(R.id.imageview);
        textView = (TextView) findViewById(R.id.textview);
        button_photo.setOnClickListener(this);
        button_vedio.setOnClickListener(this);
    }

    @Override
    public void onClick(View view) {
        switch (view.getId()){
            case R.id.photo:
                Intent choosIntent = new Intent(Intent.ACTION_GET_CONTENT);
                choosIntent.setType("image/*");
                choosIntent.addCategory(Intent.CATEGORY_OPENABLE);
                startActivityForResult(choosIntent,001);
                break;
            case R.id.vedio:
                Intent realtimeIntent = new Intent(MainActivity.this,RealTime.class);
                startActivity(realtimeIntent);

                break;
        }
    }
    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if(requestCode==001){
            try {
                bitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(),data.getData());

            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }
        long start = System.currentTimeMillis();
        detect(bitmap);
        System.out.println(System.currentTimeMillis()-start);

    }

    private boolean checkPermissionRead(){
        for (String permission : REQUIREAD_PERMISSIONS) {
            if (ContextCompat.checkSelfPermission(this, permission) != PackageManager.PERMISSION_GRANTED) {
                return false;
            }
        }
        return true;
    }
    private boolean checkPermissionCamera() {
        for (String permission : REQUIRED_PERMISSIONS) {
            if (ContextCompat.checkSelfPermission(this, permission) != PackageManager.PERMISSION_GRANTED) {
                return false;
            }
        }
        return true;
    }
    public static boolean grantPermissionRead(Activity activity) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M && activity.checkSelfPermission(
                android.Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {

            activity.requestPermissions(new String[]{
                    android.Manifest.permission.READ_EXTERNAL_STORAGE,
                    android.Manifest.permission.WRITE_EXTERNAL_STORAGE
            }, 1);
            return false;
        }
        return true;
    }
    public static boolean grantPermissionCamera(Activity activity) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M && activity.checkSelfPermission(
                android.Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {

            activity.requestPermissions(new String[]{
                    Manifest.permission.CAMERA
            }, 1);
            return false;
        }
        return true;
    }
    public  void detect(Bitmap bitmap){
        long startTime = System.currentTimeMillis();
        Bitmap first = bitmap.copy(Bitmap.Config.ARGB_8888, true);

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
        float min_face_size = 200f;
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
            System.out.println(minside);
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
            for (int i = 0; i < o_net_box.size(); i++) {
                //对图像进行人脸裁剪
                Bitmap live_bitmap = Bitmap.createBitmap(first, Math.max(o_net_box.get(i).getX1(), 0), Math.max(o_net_box.get(i).getY1(), 0), Math.min(o_net_box.get(i).getX2() - o_net_box.get(i).getX1(), first.getWidth() - Math.max(o_net_box.get(i).getX1(), 0)), Math.min(o_net_box.get(i).getY2() - o_net_box.get(i).getY1(), first.getHeight() - Math.max(o_net_box.get(i).getY1(), 0)));
                //调整尺寸为224*224
                live_bitmap = Bitmap.createScaledBitmap(live_bitmap, 224, 224, true);
                //送入人脸活体检测模型进行决策
                is_live_face = is_live(live_bitmap);
                int left, top, right, bottom;
                Canvas canvas = new Canvas(first);
                Paint paint = new Paint();
                paint.setTextSize(50);

                left = o_net_box.get(i).getX1();
                top = o_net_box.get(i).getY1();
                right = o_net_box.get(i).getX2();
                bottom = o_net_box.get(i).getY2();
                if (is_live_face == 1) {
                    paint.setColor(Color.GREEN);
                } else {
                    paint.setColor(Color.RED);
                }

                paint.setStyle(Paint.Style.STROKE);//不填充
                paint.setStrokeWidth(10);  //线的宽度
                paint.setTextSize(100);
                canvas.drawRect(left, top, right, bottom, paint);

                canvas.drawText(String.format("%.2f", depth_score), left, top - 10, paint);
            }
        }
        //输出
        long time = System.currentTimeMillis()-startTime;
        //显示在imageview上
        imageView.setImageBitmap(first);
        if(is_live_face!=0) {
            textView.setText("耗时 = " + String.valueOf(time) + "ms\nthreshold = 0.2783");
        }
        else {
            textView.setText("未检测到人脸\n无深度值\nthreshold = 0.2783");
        }
    }
    public int is_live(Bitmap bitmap){
        final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(bitmap,
                face_live_mean,face_live_std);
        final Tensor outputTensor = module_live.forward(IValue.from(inputTensor)).toTensor();
        float[] scores = outputTensor.getDataAsFloatArray();

        depth_score = get_depth_scores(scores);
        System.out.println(depth_score);
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