package org.voxime;

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

import org.voxime.asr.ModelFiles;
import org.voxime.utils.AppLog;
import org.voxime.utils.ThemeUtils;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.StandardOpenOption;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;
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
                AppLog.i(SetupActivity.this, "Setup", "voxtral download start");
                for (String name : VOXTRAL_FILES) {
                    AppLog.i(SetupActivity.this, "Setup", "downloading " + name);
                    downloadOne(name, dir);
                }
                AppLog.i(SetupActivity.this, "Setup", "voxtral download complete, files valid: " + ModelFiles.isComplete(dir));
                runOnUiThread(() -> {
                    progressBar.setVisibility(View.GONE);
                    extractedFileTV.setVisibility(View.VISIBLE);
                    extractedFileTV.setText(getString(getDownloadSuccessString()));
                    startButton.setVisibility(View.VISIBLE);
                    btn.setEnabled(true);
                    voxtralDownloading = false;
                });
            } catch (Exception e) {
                AppLog.e(SetupActivity.this, "Setup", "voxtral download failed", e);
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
        // skip files that are already complete and valid
        if (out.isFile() && out.length() == ModelFiles.expectedSize(name)) {
            runOnUiThread(() -> extractedFileTV.setText(name + " (ok)"));
            return;
        }
        long expected = ModelFiles.expectedSize(name);
        long downloaded = part.isFile() ? part.length() : 0;
        URL url = new URL(VOXTRAL_BASE + name);
        HttpURLConnection conn = (HttpURLConnection) url.openConnection();
        boolean resume = false;
        if (downloaded > 0) {
            conn.setRequestProperty("Range", "bytes=" + downloaded + "-");
            resume = true;
        }
        conn.setConnectTimeout(15000);
        conn.setReadTimeout(30000);
        int code = conn.getResponseCode();
        if (resume) {
            if (code != 206) {
                // server ignored the range: restart from scratch
                conn.disconnect();
                if (part.isFile()) part.delete();
                conn = (HttpURLConnection) url.openConnection();
                conn.setConnectTimeout(15000);
                conn.setReadTimeout(30000);
                code = conn.getResponseCode();
                downloaded = 0;
            }
        }
        if (code != 200 && code != 206) {
            conn.disconnect();
            throw new Exception("HTTP " + code + " for " + name);
        }
        long total = conn.getContentLengthLong() + downloaded;
        if (total != expected) {
            conn.disconnect();
            if (part.isFile()) part.delete();
            throw new Exception("Size mismatch for " + name + ": expected " + expected + ", got " + total);
        }
        final String shownName = name;
        runOnUiThread(() -> {
            extractedFileTV.setVisibility(View.VISIBLE);
            extractedFileTV.setText(shownName);
        });
        try (InputStream in = conn.getInputStream();
             OutputStream os = Files.newOutputStream(part.toPath(), StandardOpenOption.APPEND, StandardOpenOption.CREATE)) {
            byte[] buf = new byte[65536];
            int n;
            long lastUi = 0;
            while ((n = in.read(buf)) != -1) {
                os.write(buf, 0, n);
                downloaded += n;
                long now = System.currentTimeMillis();
                if (now - lastUi > 250) {
                    lastUi = now;
                    final int prog = (int) (downloaded * 100 / total);
                    final long done = downloaded, tot = total;
                    runOnUiThread(() -> {
                        progressBar.setProgress(prog);
                        extractedFileTV.setText(shownName + "  " + done / 1048576 + "/" + tot / 1048576 + " MB");
                    });
                }
            }
        } finally {
            conn.disconnect();
        }
        if (downloaded != expected) {
            throw new Exception("Incomplete download of " + name + ": " + downloaded + "/" + expected);
        }
        // atomic-ish activation: only replace after size validated
        if (out.isFile()) out.delete();
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
            try {
                extractZipSafe(context, targetDir, zipFile);
                runOnUiThread(() -> {
                    progressBar.setIndeterminate(false);
                    progressBar.setVisibility(View.GONE);
                    extractedFileTV.setVisibility(View.VISIBLE);
                    extractedFileTV.setText(getString(getResources().getIdentifier("download_voxtral_success", "string", getPackageName())));
                    startButton.setVisibility(View.VISIBLE);
                });
            } catch (final Exception e) {
                Log.e("SetupActivity", "ZIP import failed", e);
                runOnUiThread(() -> {
                    progressBar.setIndeterminate(false);
                    progressBar.setVisibility(View.GONE);
                    Toast.makeText(this, getString(getResources().getIdentifier("download_voxtral_failed", "string", getPackageName())) + ": " + e.getMessage(), Toast.LENGTH_LONG).show();
                });
            }
        });
        thread.start();
    }

    /**
     * Hardened ZIP import: only the seven expected flat file names are accepted,
     * each entry is validated by size, nothing is written outside the model dir.
     */
    private void extractZipSafe(Context context, File targetDir, Uri zipFile) throws Exception {
        if (targetDir == null) throw new Exception("storage unavailable");
        String canonicalBase = targetDir.getCanonicalPath() + File.separator;
        long totalLimit = ModelFiles.totalSize() * 2;
        long writtenTotal = 0;
        java.util.Set<String> seen = new java.util.HashSet<>();
        try (InputStream src = context.getContentResolver().openInputStream(zipFile);
             ZipInputStream zis = new ZipInputStream(src)) {
            ZipEntry entry;
            byte[] buf = new byte[65536];
            while ((entry = zis.getNextEntry()) != null) {
                String name = entry.getName();
                if (entry.isDirectory()) throw new Exception("unexpected directory in zip: " + name);
                if (name.contains("/") || name.contains("\\") || !ModelFiles.isExpectedName(name)) {
                    throw new Exception("unexpected file in zip: " + name);
                }
                File outFile = new File(targetDir, name);
                if (!outFile.getCanonicalPath().startsWith(canonicalBase)) {
                    throw new Exception("path traversal blocked: " + name);
                }
                File partFile = new File(targetDir, name + ".tmp");
                long expected = ModelFiles.expectedSize(name);
                long written = 0;
                try (OutputStream os = Files.newOutputStream(partFile.toPath(), StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING)) {
                    int n;
                    while ((n = zis.read(buf)) != -1) {
                        written += n;
                        writtenTotal += n;
                        if (written > expected || writtenTotal > totalLimit) {
                            throw new Exception("zip too large: " + name);
                        }
                        os.write(buf, 0, n);
                    }
                }
                if (written != expected) {
                    partFile.delete();
                    throw new Exception("size mismatch for " + name + ": " + written + "/" + expected);
                }
                if (outFile.isFile()) outFile.delete();
                if (!partFile.renameTo(outFile)) throw new Exception("rename failed: " + name);
                seen.add(name);
                final String shown = name;
                runOnUiThread(() -> extractedFileTV.setText(shown));
            }
        }
        List<String> missing = new ArrayList<>();
        for (ModelFiles.Entry e : ModelFiles.REQUIRED) {
            if (!seen.contains(e.name)) missing.add(e.name);
        }
        if (!missing.isEmpty()) {
            throw new Exception("zip is missing: " + android.text.TextUtils.join(", ", missing));
        }
    }
}
