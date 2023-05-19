package com.example.mtcnn;

public class Box {
    public int getX1() {
        return x1;
    }
    public void setX1(int x1) {
        this.x1 = x1;
    }
    public int getY1() {
        return y1;
    }
    public void setY1(int y1) {
        this.y1 = y1;
    }
    public int getX2() {
        return x2;
    }
    public void setX2(int x2) {
        this.x2 = x2;
    }
    public int getY2() {
        return y2;
    }
    public void setY2(int y2) {
        this.y2 = y2;
    }
    public float getCls() {
        return cls;
    }
    public void setCls(float cls) {
        this.cls = cls;
    }
    public int getPx1() {
        return px1;
    }
    public void setPx1(int px1) {
        this.px1 = px1;
    }
    public int getPy1() {
        return py1;
    }
    public void setPy1(int py1) {
        this.py1 = py1;
    }
    public int getPx2() {
        return px2;
    }
    public void setPx2(int px2) {
        this.px2 = px2;
    }
    public int getPy2() {
        return py2;
    }
    public void setPy2(int py2) {
        this.py2 = py2;
    }
    public int getPx3() {
        return px3;
    }
    public void setPx3(int px3) {
        this.px3 = px3;
    }
    public int getPy3() {
        return py3;
    }
    public void setPy3(int py3) {
        this.py3 = py3;
    }
    public int getPx4() {
        return px4;
    }
    public void setPx4(int px4) {
        this.px4 = px4;
    }
    public int getPy4() {
        return py4;
    }
    public void setPy4(int py4) {
        this.py4 = py4;
    }
    public int getPx5() {
        return px5;
    }
    public void setPx5(int px5) {
        this.px5 = px5;
    }
    public int getPy5() {
        return py5;
    }
    public void setPy5(int py5) {
        this.py5 = py5;
    }
    private int x1,y1,x2,y2;
    private float cls;
    @Override
    public String toString() {
        return "Box{" +
                "x1=" + x1 +
                ", y1=" + y1 +
                ", x2=" + x2 +
                ", y2=" + y2 +
                ", cls=" + cls +
                ", px1=" + px1 +
                ", py1=" + py1 +
                ", px2=" + px2 +
                ", py2=" + py2 +
                ", px3=" + px3 +
                ", py3=" + py3 +
                ", px4=" + px4 +
                ", py4=" + py4 +
                ", px5=" + px5 +
                ", py5=" + py5 +
                '}';
    }
    private int px1,py1,px2,py2,px3,py3,px4,py4,px5,py5;
}
