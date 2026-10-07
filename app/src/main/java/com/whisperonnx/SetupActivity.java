package com.whisperonnx;

import static android.content.Intent.FLAG_ACTIVITY_NEW_TASK;

import android.content.Context;
import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.app.AppCompatDelegate;

import com.whisperonnx.utils.ThemeUtils;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.StandardOpenOption;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

public class SetupActivity extends AppCompatActivity {
    ActivityResultLauncher<Intent> install;
    ProgressBar progressBar;
    TextView extractedFileTV;
    Button startButton;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_download);
        ThemeUtils.setStatusBarAppearance(this);
        AppCompatDelegate.setDefaultNightMode(AppCompatDelegate.MODE_NIGHT_FOLLOW_SYSTEM);
        progressBar = findViewById(R.id.progress_bar);
        extractedFileTV = findViewById(R.id.extracted_file);
        startButton = findViewById(R.id.button_start);

        File sdcardDataFolder = getExternalFilesDir(null);

        install = registerForActivityResult(
                new ActivityResultContracts.StartActivityForResult(),
                result -> {
                    if (result.getData()!=null && result.getData().getData()!=null) zipExtract(this, sdcardDataFolder, result.getData().getData());
                });

    }

     public void downloadModel(View v){
         Toast.makeText(this,"Download",Toast.LENGTH_SHORT).show();
         startActivity(new Intent(Intent.ACTION_VIEW, Uri.parse("https://huggingface.co/DocWolle/whisperOnnx/blob/main/whisper_small_int8.zip")));
     }

    private static final String[] VOXTRAL_FILES = {
            "audio_encoder_q4f16.onnx", "audio_encoder_q4f16.onnx_data",
            "embed_tokens_q4.onnx", "embed_tokens_q4.onnx_data",
            "decoder_model_merged_q4.onnx", "decoder_model_merged_q4.onnx_data", "decoder_model_merged_q4.onnx_data_1"
    };
    private static final String VOXTRAL_BASE = "https://huggingface.co/onnx-community/Voxtral-Mini-3B-2507-ONNX/resolve/main/onnx/";
    private boolean voxtralDownloading = false;

    public void downloadVoxtral(View v) {
        if (voxtralDownloading) return;
        File dir = getExternalFilesDir(null);
        if (dir == null) { Toast.makeText(this, "Storage unavailable", Toast.LENGTH_SHORT).show(); return; }
        voxtralDownloading = true;
        progressBar.setVisibility(View.VISIBLE);
        progressBar.setIndeterminate(false);
        progressBar.setProgress(0);
        Button btn = findViewById(R.id.download_voxtral_button);
        btn.setEnabled(false);
        Thread t = new Thread(() -> {
            try {
                for (String name : VOXTRAL_FILES) {
                    downloadOne(name, dir);
                }
                runOnUiThread(() -> {
                    progressBar.setVisibility(View.GONE);
                    extractedFileTV.setVisibility(View.VISIBLE);
                    extractedFileTV.setText(getString(getDownloadSuccessString()));
                    startButton.setVisibility(View.VISIBLE);
                    btn.setEnabled(true);
                    voxtralDownloading = false;
                });
            } catch (Exception e) {
                Log.e("SetupActivity", "Voxtral download failed", e);
                runOnUiThread(() -> {
                    progressBar.setVisibility(View.GONE);
                    btn.setEnabled(true);
                    voxtralDownloading = false;
                    Toast.makeText(this, getString(getDownloadFailedString()) + ": " + e.getMessage(), Toast.LENGTH_LONG).show();
                });
            }
        });
        t.start();
    }

    private int getDownloadSuccessString() {
        return getResources().getIdentifier("download_voxtral_success", "string", getPackageName());
    }

    private int getDownloadFailedString() {
        return getResources().getIdentifier("download_voxtral_failed", "string", getPackageName());
    }

    private void downloadOne(String name, File dir) throws Exception {
        File out = new File(dir, name);
        File part = new File(dir, name + ".part");
        URL url = new URL(VOXTRAL_BASE + name);
        long downloaded = part.exists() ? part.length() : 0;
        HttpURLConnection conn = (HttpURLConnection) url.openConnection();
        if (downloaded > 0) conn.setRequestProperty("Range", "bytes=" + downloaded + "-");
        conn.setConnectTimeout(15000);
        conn.setReadTimeout(30000);
        int code = conn.getResponseCode();
        if (code != 200 && code != 206) throw new Exception("HTTP " + code + " for " + name);
        long total = conn.getContentLengthLong() + downloaded;
        final String shownName = name;
        runOnUiThread(() -> {
            extractedFileTV.setVisibility(View.VISIBLE);
            extractedFileTV.setText(shownName);
        });
        try (InputStream in = conn.getInputStream(); OutputStream os = Files.newOutputStream(part.toPath(), StandardOpenOption.APPEND, StandardOpenOption.CREATE)) {
            byte[] buf = new byte[65536];
            int n;
            long lastUi = 0;
            while ((n = in.read(buf)) != -1) {
                os.write(buf, 0, n);
                downloaded += n;
                long now = System.currentTimeMillis();
                if (now - lastUi > 250 && total > 0) {
                    lastUi = now;
                    final int prog = (int) (downloaded * 100 / total);
                    final long done = downloaded, tot = total;
                    runOnUiThread(() -> {
                        progressBar.setProgress(prog);
                        extractedFileTV.setText(shownName + "  " + done / 1048576 + "/" + tot / 1048576 + " MB");
                    });
                }
            }
        }
        if (out.exists()) out.delete();
        if (!part.renameTo(out)) throw new Exception("rename failed: " + name);
    }
    public void installModel(View v){
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.setType("application/zip");
        install.launch(intent);

    }

    public void startMain(View v){
        Intent intent = new Intent(this, MainActivity.class);
        intent.addFlags(FLAG_ACTIVITY_NEW_TASK);
        startActivity(intent);
    }

    public void zipExtract(Context context, File targetDir, Uri zipFile) {
        progressBar.setVisibility(View.VISIBLE);
        progressBar.setIndeterminate(true);
        Thread thread = new Thread(() -> {
            ZipEntry zipEntry;
            int readLen;
            byte[] readBuffer = new byte[4096];
            try {
                InputStream src = context.getContentResolver().openInputStream(zipFile);
                try {
                    try (ZipInputStream zipInputStream = new ZipInputStream(src)) {
                        while ((zipEntry = zipInputStream.getNextEntry()) != null) {
                            File extractedFile = new File(targetDir ,zipEntry.getName());
                            runOnUiThread(()->{
                                extractedFileTV.setVisibility(View.VISIBLE);
                                extractedFileTV.setText(extractedFile.getName());
                            });
                            try (OutputStream outputStream = Files.newOutputStream(extractedFile.toPath())) {
                                while ((readLen = zipInputStream.read(readBuffer)) != -1) {
                                    outputStream.write(readBuffer, 0, readLen);
                                }
                            }
                        }
                        runOnUiThread(()->{
                            progressBar.setIndeterminate(false);
                            progressBar.setVisibility(View.GONE);
                            extractedFileTV.setVisibility(View.GONE);
                            startButton.setVisibility(View.VISIBLE);
                        });
                    }
                } catch (IOException ioException) {
                    ioException.printStackTrace();
                }
            } catch (FileNotFoundException e) {
                e.printStackTrace();
            }
        });
        thread.start();
    }
}
