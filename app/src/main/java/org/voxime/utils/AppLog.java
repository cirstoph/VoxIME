package org.voxime.utils;

import android.content.Context;
import android.util.Log;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;

/**
 * File logger for on-device diagnosis. Writes voxime.log into the app's
 * external files dir (same folder as the model), so the user can share it
 * from within the app. Also mirrors to logcat.
 *
 * Keep it tiny and exception-safe: logging must never crash the app.
 */
public final class AppLog {

    private static final String FILE_NAME = "voxime.log";
    private static final long MAX_BYTES = 2 * 1024 * 1024;   // rotate at 2 MB
    private static final Object LOCK = new Object();
    private static final BlockingQueue<String> PENDING = new ArrayBlockingQueue<>(1024);
    private static volatile boolean writerStarted = false;
    private static File logFile;

    private AppLog() {}

    public static File getLogFile(Context context) {
        init(context);
        return logFile;
    }

    private static void init(Context context) {
        if (logFile != null) return;
        try {
            File dir = context.getExternalFilesDir(null);
            if (dir == null) return;
            logFile = new File(dir, FILE_NAME);
        } catch (Exception ignored) {
        }
    }

    public static void i(Context context, String tag, String message) {
        write(context, "I", tag, message, null);
    }

    public static void w(Context context, String tag, String message) {
        write(context, "W", tag, message, null);
    }

    public static void e(Context context, String tag, String message, Throwable t) {
        write(context, "E", tag, message, t);
    }

    /** Explicitly used by components without a Context handy (already initialized). */
    public static void e(String tag, String message, Throwable t) {
        write(null, "E", tag, message, t);
    }

    /**
     * Async file logging: the caller only formats the line and hands it to the
     * background writer; no FileWriter on the inference/UI thread. Errors
     * mirror to logcat synchronously and are never lost by a dropped queue
     * entry (errors are rare, the queue covers bursts of debug lines).
     */
    private static void write(Context context, String level, String tag, String message, Throwable t) {
        if (context != null) init(context);
        Log.d(tag, message);
        if ("E".equals(level)) {
            writeLine(level, tag, message, t);
            return;
        }
        String ts = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS", Locale.US).format(new Date());
        String line = ts + ' ' + level + '/' + tag + ": " + message + '\n';
        if (!PENDING.offer(line)) {
            writeLine(level, tag, message, t);   // queue full: rare, write directly
        }
        startWriter();
    }

    private static synchronized void startWriter() {
        if (writerStarted) return;
        writerStarted = true;
        Thread writer = new Thread(() -> {
            while (true) {
                try {
                    String line = PENDING.take();
                    synchronized (LOCK) {
                        if (logFile != null) {
                            try (FileWriter fw = new FileWriter(logFile, true)) {
                                fw.append(line);
                            }
                            rotateIfNeeded();
                        }
                    }
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    return;
                } catch (Exception ignored) {
                }
            }
        }, "voxime-logwriter");
        writer.setDaemon(true);
        writer.start();
    }

    private static void writeLine(String level, String tag, String message, Throwable t) {
        synchronized (LOCK) {
            if (logFile == null) return;
            try {
                rotateIfNeeded();
                String ts = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS", Locale.US).format(new Date());
                StringBuilder sb = new StringBuilder(ts).append(' ').append(level).append('/').append(tag).append(": ").append(message);
                if (t != null) {
                    StringWriter sw = new StringWriter();
                    t.printStackTrace(new PrintWriter(sw));
                    sb.append("\n").append(sw);
                }
                sb.append('\n');
                try (FileWriter fw = new FileWriter(logFile, true)) {
                    fw.append(sb);
                }
            } catch (IOException ignored) {
            }
        }
    }

    private static void rotateIfNeeded() {
        try {
            if (logFile.exists() && logFile.length() > MAX_BYTES) {
                File old = new File(logFile.getParentFile(), FILE_NAME + ".1");
                if (old.exists()) old.delete();
                logFile.renameTo(old);
            }
        } catch (Exception ignored) {
        }
    }

    /** Clears the log; called right before a dictation test if the user wants a clean file. */
    public static void clear(Context context) {
        init(context);
        synchronized (LOCK) {
            try { if (logFile != null) logFile.delete(); } catch (Exception ignored) {}
        }
    }
}
