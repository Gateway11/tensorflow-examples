package com.vsys.voicerec.example;

import android.app.Activity;
import android.os.Bundle;


public class MainActivity extends Activity {
	
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        
    }
    
    private native int native_init();

    private native int native_release();

    private native int native_process();
}
