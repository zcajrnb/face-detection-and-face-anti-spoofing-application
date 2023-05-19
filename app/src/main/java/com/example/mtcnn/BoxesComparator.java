package com.example.mtcnn;

import java.util.Comparator;

public class BoxesComparator implements Comparator<Box> {
    @Override
    public int compare(Box o1, Box o2) {
        if(o1.getCls()<=o2.getCls()){
            return 1;
        }
        else {
            return -1;
        }
    }
}
