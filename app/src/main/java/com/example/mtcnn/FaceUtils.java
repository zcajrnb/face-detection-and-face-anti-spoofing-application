package com.example.mtcnn;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Matrix;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class FaceUtils {
    public static String assetFilePath(Context context, String assetName) throws IOException {
        File file = new File(context.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }
        try (InputStream is = context.getAssets().open(assetName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        }
    }
    public static int maximum(int a,int b){
        if(a>=b){
            return a;
        }
        else {
            return b;
        }
    }
    public static int minmum(int a,int b){
        if(a<=b){
            return a;
        }
        else {
            return b;
        }
    }
    public static List<Integer> nonzero(float facecls[], float threadshold){
        List<Integer> index = new ArrayList<>();
        for(int i = 0;i < facecls.length;i++){
            if(facecls[i]>threadshold){
                index.add(i);
            }
        }
        return index;
    }
    public static List<Float> getCls(List<Integer> loaction,float facecls[]){
        List<Float> face = new ArrayList<>();
        if (loaction.size()==0){
            return face;
        }
        for (int i = 0; i < loaction.size(); i++) {
            face.add(facecls[loaction.get(i)]);
        }
        return face;
    }
    public static List<Offset> getOffset(List<Integer> loaction,int cls_length,float[] facebox){
        List<Offset> face = new ArrayList<>();
        if (loaction.size()==0){
            return face;
        }
        for(int i=0;i<loaction.size();i++) {
            Offset offset = new Offset();
            offset.setX1(facebox[loaction.get(i)+cls_length*0]);
            offset.setY1(facebox[loaction.get(i)+cls_length*1]);
            offset.setX2(facebox[loaction.get(i)+cls_length*2]);
            offset.setY2(facebox[loaction.get(i)+cls_length*3]);
            face.add(offset);
        }
        return face;
    }
    public static List<Box> transebox(int width,List<Integer> loaction,List<Float> facecls,List<Offset> facebox,float scale,int stride,int side_len){
        List<Box> list = new ArrayList<>();
        int index0,index1;
        for(int i=0;i<facecls.size();i++){
            Box bx = new Box();
            index0 = loaction.get(i)/width;
            index1 = loaction.get(i)%width;
            int _x1 = (int)((index1*stride)/scale);
            int _y1 = (int)((index0*stride)/scale);
            int _x2 = (int)((index1*stride+side_len)/scale);
            int _y2 = (int)((index0*stride+side_len)/scale);
            int w_side = _x2 - _x1;
            int h_side = _y2-_y1;

            bx.setX1((int)(_x1+w_side*facebox.get(i).getX1()));
            bx.setY1((int)(_y1+h_side*facebox.get(i).getY1()));
            bx.setX2((int)(_x2+w_side*facebox.get(i).getX2()));
            bx.setY2((int)(_y2+h_side*facebox.get(i).getY2()));
            bx.setCls(facecls.get(i));
            list.add(bx);
        }
        return list;
    }
    public static float iou(Box box1,Box box2,boolean isMin){
        int area1 = (box1.getX2()-box1.getX1())*(box1.getY2()-box1.getY1());
        int area2 = (box2.getX2()-box2.getX1())*(box2.getY2()-box2.getY1());
        int x1 = maximum(box1.getX1(),box2.getX1());
        int y1 = maximum(box1.getY1(),box2.getY1());
        int x2 = minmum(box1.getX2(),box2.getX2());
        int y2 = minmum(box1.getY2(),box2.getY2());
        int w = maximum(0,x2-x1);
        int h = maximum(0,y2-y1);

        int inter = w*h;

        if(isMin==true){
            return (1.0f*inter)/(1.0f*minmum(area1,area2));
        }
        else {
            return (1.0f*inter)/(1.0f*(area1+area2-inter));
        }
    }
    public static List<Box> nms(List<Box> boxes,float threadshold,boolean isMin){
        List<Box> r_box = new ArrayList<>();

        List<Box> b_box = new ArrayList<>();
        if (boxes.size()==0){
            return r_box;
        }
        BoxesComparator comparator = new BoxesComparator();

        while (boxes.size()>1){
            Collections.sort(boxes,comparator);

            Box a_box = new Box();
            a_box = boxes.get(0);
            r_box.add(a_box);

            for(int i=1;i<boxes.size();i++){
                if(iou(a_box,boxes.get(i),isMin)<threadshold){
                    b_box.add(boxes.get(i));
                }
            }
            boxes.clear();
            boxes.addAll(b_box);
            b_box.clear();
        }
        if(boxes.size()>0){
            r_box.add(boxes.get(0));
        }
        return r_box;
    }
    public static List<Box> convert_to_square(List<Box> boxes,int width,int height){
        List<Box> square_box = new ArrayList<>();
        if (boxes.size()==0){
            return square_box;
        }
        int w,h;
        int x=0,y=0;
        for(int i=0;i<boxes.size();i++){
            Box bx = new Box();
            w = boxes.get(i).getX2()-boxes.get(i).getX1()+1;
            h = boxes.get(i).getY2()-boxes.get(i).getY1()+1;
            int max_side = maximum(w,h);
            bx.setX1(boxes.get(i).getX1()+w/2-max_side/2);
            bx.setY1(boxes.get(i).getY1()+h/2-max_side/2);
            bx.setX2(bx.getX1()+max_side-1);
            bx.setY2(bx.getY1()+max_side-1);
            bx.setCls(boxes.get(i).getCls());
            if(bx.getX1()<0){
                x = bx.getX1();
                bx.setX1(0);
                bx.setX2(bx.getX2()-x);
            }
            if(bx.getY1()<0){
                y = bx.getY1();
                bx.setY1(0);
                bx.setY2(bx.getY2()-y);
            }
            if(bx.getX2()>width){
                x = bx.getX2();
                bx.setX2(width);
                bx.setX1(bx.getX1()-x);
            }
            if(bx.getY2()>height){
                y = bx.getY2();
                bx.setY2(height);
                bx.setY1(bx.getY1()-y);
            }
            if(bx.getX1()<0||bx.getY1()<0||bx.getX2()>width||bx.getY2()>height){
                continue;
            }
            square_box.add(bx);

        }
        return square_box;
    }
    public static Bitmap crop_and_resize(Bitmap bitmap, Box bx, int size){
        //(2)crop and resize
        Matrix matrix = new Matrix();
        float scale=1.0f*size/(bx.getX2()-bx.getX1());
        matrix.postScale(scale, scale);
        Bitmap croped=Bitmap.createBitmap(bitmap, bx.getX1(),bx.getY1(),bx.getX2()-bx.getX1(),bx.getY2()-bx.getY1(),matrix,true);
        return croped;
    }
    public static List<Box> zhenghernetbox(List<Box> p_r_box,List<Float> facecls,List<Offset> facebox){
        List<Box> list = new ArrayList<>();
        for(int i=0;i<facecls.size();i++){
            Box bx = new Box();
            int _x1 = p_r_box.get(i).getX1();
            int _y1 = p_r_box.get(i).getY1();
            int _x2 = p_r_box.get(i).getX2();
            int _y2 = p_r_box.get(i).getY2();
            int w_side = _x2 - _x1;
            int h_side = _y2-_y1;

            bx.setX1((int)(_x1+w_side*facebox.get(i).getX1()));
            bx.setY1((int)(_y1+h_side*facebox.get(i).getY1()));
            bx.setX2((int)(_x2+w_side*facebox.get(i).getX2()));
            bx.setY2((int)(_y2+h_side*facebox.get(i).getY2()));
            bx.setCls(facecls.get(i));
            list.add(bx);
        }
        return list;
    }
    public static List<Point> getPoint(List<Integer> loaction,float[] facepoint){
        List<Point> p = new ArrayList<>();
        if (loaction.size()==0){
            return p;
        }
        for(int i=0;i<loaction.size();i++){
            Point point = new Point();
            point.setPx1(facepoint[loaction.get(i)]);
            point.setPy1(facepoint[loaction.get(i)+1]);
            point.setPx2(facepoint[loaction.get(i)+2]);
            point.setPy2(facepoint[loaction.get(i)+3]);
            point.setPx3(facepoint[loaction.get(i)+4]);
            point.setPy3(facepoint[loaction.get(i)+5]);
            point.setPx4(facepoint[loaction.get(i)+6]);
            point.setPy4(facepoint[loaction.get(i)+7]);
            point.setPx5(facepoint[loaction.get(i)+8]);
            point.setPy5(facepoint[loaction.get(i)+9]);
            p.add(point);
        }
        return p;
    }
    public static List<Box> zhengheonetbox(List<Box> r_o_box,List<Float> facecls,List<Offset> facebox,List<Point> facepoint){
        List<Box> list = new ArrayList<>();
        for(int i=0;i<facecls.size();i++){
            Box bx = new Box();
            int _x1 = r_o_box.get(i).getX1();
            int _y1 = r_o_box.get(i).getY1();
            int _x2 = r_o_box.get(i).getX2();
            int _y2 = r_o_box.get(i).getY2();
            int w_side = _x2 - _x1;
            int h_side = _y2-_y1;
            bx.setX1((int)(_x1+w_side*facebox.get(i).getX1()));
            bx.setY1((int)(_y1+h_side*facebox.get(i).getY1()));
            bx.setX2((int)(_x2+w_side*facebox.get(i).getX2()));
            bx.setY2((int)(_y2+h_side*facebox.get(i).getY2()));
            bx.setCls(facecls.get(i));
            bx.setPx1((int)(_x1+w_side*facepoint.get(i).getPx1()));
            bx.setPy1((int)(_y1+h_side*facepoint.get(i).getPy1()));
            bx.setPx2((int)(_x1+w_side*facepoint.get(i).getPx2()));
            bx.setPy2((int)(_y1+h_side*facepoint.get(i).getPy2()));
            bx.setPx3((int)(_x1+w_side*facepoint.get(i).getPx3()));
            bx.setPy3((int)(_y1+h_side*facepoint.get(i).getPy3()));
            bx.setPx4((int)(_x1+w_side*facepoint.get(i).getPx4()));
            bx.setPy4((int)(_y1+h_side*facepoint.get(i).getPy4()));
            bx.setPx5((int)(_x1+w_side*facepoint.get(i).getPx5()));
            bx.setPy5((int)(_y1+h_side*facepoint.get(i).getPy5()));
            list.add(bx);
        }
        return list;
    }

}
