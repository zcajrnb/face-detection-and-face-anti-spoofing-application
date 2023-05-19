package com.example.mtcnn;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.util.AttributeSet;
import android.view.View;

import java.util.List;


public class FaceBoxView extends View {
    Bitmap bitmap;
    int[] is_live;
    float[] depth_score;
    private Paint mPaintRectangle;
    private List<Box> mResults=null;
    public FaceBoxView(Context context){ super(context);}
    public FaceBoxView(Context context, AttributeSet attrs){
        super(context, attrs);
        mPaintRectangle = new Paint();
        mPaintRectangle.setColor(Color.RED);

    }
    @Override
    protected void onDraw(Canvas canvas){
        super.onDraw(canvas);
        if (mResults==null) return;
        for(int i=0;i<mResults.size();i++){
            float off_x = (bitmap.getWidth()-getWidth())/2.0f;
            float off_y = (bitmap.getHeight()-getHeight())/2.0f;
            float left = mResults.get(i).getX1()-off_x;
            float top = mResults.get(i).getY1()-off_y ;
            float right = mResults.get(i).getX2()-off_x;
            float bottom = mResults.get(i).getY2()-off_y;
            mPaintRectangle.setStrokeWidth(10);
            mPaintRectangle.setStyle(Paint.Style.STROKE);
            if(is_live[i]==1){
                mPaintRectangle.setColor(Color.GREEN);
            }else{
                mPaintRectangle.setColor(Color.RED);
            }
            canvas.drawRect(new RectF(left,top,right,bottom), mPaintRectangle);
            mPaintRectangle.setTextSize(50);
            canvas.drawText(String.format("%.2f",depth_score[i]),left,top-10,mPaintRectangle);
        }
    }
    public void setResults(List<Box> results, Bitmap bitmap, int[] is_live,float[] depth_score ){

        this.is_live = is_live;
        this.bitmap = bitmap;
        this.mResults = results;
        this.depth_score = depth_score;
        invalidate();}
}
