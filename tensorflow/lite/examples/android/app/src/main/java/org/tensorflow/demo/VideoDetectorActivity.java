/**
 * @author poioit
 * @date 2018/12/27
 */

package org.tensorflow.demo;

import android.Manifest;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.pm.ActivityInfo;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.media.MediaMetadataRetriever;
import android.media.MediaPlayer;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.support.annotation.Nullable;
import android.support.v7.app.AlertDialog;
import android.support.v7.app.AppCompatActivity;
import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Paint.Style;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.SystemClock;
import android.util.Log;
import android.util.Size;
import android.util.TypedValue;
import android.view.Gravity;
import android.view.View;
import android.view.ViewTreeObserver;
import android.view.Window;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.MediaController;
import android.widget.Toast;
import android.widget.VideoView;
import android.view.ViewGroup.LayoutParams;

import org.tensorflow.demo.OverlayView.DrawCallback;
import org.tensorflow.demo.env.BorderedText;
import org.tensorflow.demo.env.ImageUtils;
import org.tensorflow.demo.env.Logger;
import org.tensorflow.demo.tracking.MultiBoxTracker;
import org.tensorflow.lite.demo.R;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.LinkedList;
import java.util.List;
import java.util.Vector;

import static android.media.MediaMetadataRetriever.OPTION_CLOSEST_SYNC;
import static android.media.MediaMetadataRetriever.OPTION_NEXT_SYNC;

/**
 * An activity that uses a TensorFlowMultiBoxDetector and ObjectTracker to detect and then track
 * objects.
 */
/**
 * VideoDetectorActivity class
 *
 * @author poioit
 * @date 2018/12/27
 */
public class VideoDetectorActivity extends AppCompatActivity {
    private static final Logger LOGGER = new Logger();
    private static final String TAG = "HANK_DEBUG@" + VideoDetectorActivity.class.getSimpleName();
    private final int CHOOSE_IMAGE = 1001;
    private final int CHOOSE_VIDEO = 1002;
    private final int CHOOSE_CAMERA = 1003;

    private static final int PERMISSIONS_REQUEST = 1;

    private static final String PERMISSION_CAMERA = Manifest.permission.CAMERA;
    private static final String PERMISSION_STORAGE = Manifest.permission.WRITE_EXTERNAL_STORAGE;


    // Configuration values for the prepackaged SSD model.
    private static final int TF_OD_API_INPUT_SIZE = 300;
    private static final boolean TF_OD_API_IS_QUANTIZED = true;
    private static final String TF_OD_API_MODEL_FILE = "detect.tflite";
    private static final String TF_OD_API_LABELS_FILE = "coco_labels_list.txt";
    private boolean debug = false;
    private Handler handler;
    private HandlerThread handlerThread;
    private Runnable postInferenceCallback;
    private Runnable imageConverter;
    private Button btnCamera;
    private VideoView videoView;
    private FrameLayout flayout;
    private MediaMetadataRetriever mediaMetadataRetriever;
    private MediaController myMediaController;
    private Handler mHandlerTime = new Handler();

    private byte[][] yuvBytes = new byte[3][];
    private int[] rgbBytes = null;
    private int yRowStride;

    protected int previewWidth = 0;
    protected int previewHeight = 0;
    private boolean initial_tracker = false;

    private int nTime = 0;

    // Which detection model to use: by default uses Tensorflow Object Detection API frozen
    // checkpoints.
    private enum DetectorMode {
        TF_OD_API;
    }

    private static final DetectorMode MODE = DetectorMode.TF_OD_API;

    // Minimum detection confidence to track a detection.
    private static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.55f;

    private static final boolean MAINTAIN_ASPECT = false;

    private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 480);

    private static final boolean SAVE_PREVIEW_BITMAP = false;
    private static final float TEXT_SIZE_DIP = 10;
    private long regularPollingTime = 0;
    private long initialPollingTime = 200L;
    private long retryPollingTime = 50L;
    private boolean stopPlaying = false;

    private Integer sensorOrientation;

    private Classifier detector;

    private long lastProcessingTimeMs;
    private Bitmap rgbFrameBitmap = null;
    private Bitmap croppedBitmap = null;
    private Bitmap cropCopyBitmap = null;

    private boolean computingDetection = false;

    private long timestamp = 0;

    private Matrix frameToCropTransform;
    private Matrix cropToFrameTransform;

    private MultiBoxTracker tracker;

    private byte[] luminanceCopy;

    private BorderedText borderedText;

    private boolean hasPermission() {
        {
            return checkSelfPermission(PERMISSION_CAMERA) == PackageManager.PERMISSION_GRANTED &&
                    checkSelfPermission(PERMISSION_STORAGE) == PackageManager.PERMISSION_GRANTED;
        }
    }

    private void requestPermission() {
        {
            if (shouldShowRequestPermissionRationale(PERMISSION_CAMERA) ||
                    shouldShowRequestPermissionRationale(PERMISSION_STORAGE)) {
                Toast.makeText(VideoDetectorActivity.this,
                        "Camera AND storage permission are required for this demo", Toast.LENGTH_LONG).show();
            }
            requestPermissions(new String[] {PERMISSION_CAMERA, PERMISSION_STORAGE}, PERMISSIONS_REQUEST);
        }
    }

    @Nullable
    public final String getPath(Uri uri) {

        String[] projection = new String[]{"_data"};
        Cursor cursor = this.getContentResolver().query(uri, projection, (String)null, (String[])null, (String)null);
        if (cursor != null) {
            int column_index = cursor.getColumnIndexOrThrow("_data");
            cursor.moveToFirst();
            return cursor.getString(column_index);
        } else {
            return null;
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode == 0) {
            Log.d("what", "cancle");
        } else {

            if (requestCode == this.CHOOSE_VIDEO) {
                if (data != null) {
                    Uri contentURI = data.getData();
                    String selectedVideoPath = getPath(contentURI);
                    Log.d("hank_debug", selectedVideoPath);

                    //videoView.setZOrderMediaOverlay(true);
                    //videoView.setZOrderOnTop(true);
                    videoView.setVideoURI(contentURI);
                    //videoView.setRotation(90);
                    videoView.requestFocus();
                    videoView.start();

                    mediaMetadataRetriever.setDataSource(String.valueOf(selectedVideoPath));
                    myMediaController = new MediaController((Context)this);
                    videoView.setMediaController(this.myMediaController);
                    Log.i( "hank_debug", "vv height:" + videoView.getMeasuredHeight() + " width:" + videoView.getMeasuredWidth());
                    previewHeight = videoView.getMeasuredHeight();
                    previewWidth = videoView.getMeasuredWidth();
                    if( !initial_tracker ) {
                        setupTracker();
                        initial_tracker = true;
                    }
                    MediaPlayer.OnCompletionListener myVideoViewCompletionListener =
                            new MediaPlayer.OnCompletionListener() {

                                @Override
                                public void onCompletion(MediaPlayer arg0) {
                                    Toast.makeText(VideoDetectorActivity.this, "End of Video",
                                            Toast.LENGTH_LONG).show();


                                    btnCamera.setVisibility(View.VISIBLE);
                                    stopPlaying = true;
                                }
                            };
                    videoView.setOnCompletionListener(myVideoViewCompletionListener);
                    btnCamera.setVisibility(View.INVISIBLE);

                    stopPlaying = false;
                    mHandlerTime.postDelayed((Runnable)this.timerRun, initialPollingTime);
                }
            } else if (requestCode == this.CHOOSE_CAMERA) {
                Uri contentURI = data.getData();
                String selectedVideoPath = getPath(contentURI);
                Log.d("path", selectedVideoPath);
            }

        }
    }

    private Runnable timerRun=new Runnable () {
        @Override
        public void run() {
            if( computingDetection )
            {
                Log.i("hank_debug", "re-entrance skip");
                trackingOverlay.postInvalidate();
                mHandlerTime.postDelayed(this, retryPollingTime);

                return;
            }
            Log.i( "hank_debug", " " + videoView.isPlaying());
            if( stopPlaying) {
                final List<Classifier.Recognition> mappedRecognitions =
                        new LinkedList<Classifier.Recognition>();
                long currentPosition = videoView.getCurrentPosition();
                tracker.trackResults(mappedRecognitions, luminanceCopy, currentPosition);

                trackingOverlay.postInvalidate();
                return;
            }
            computingDetection = true;
            ++nTime; // 經過的秒數 + 1
            long currentPosition = videoView.getCurrentPosition();
            byte[] originalLuminance = getLuminance();


            //unit in microsecond
            Bitmap bmFrame = mediaMetadataRetriever.getFrameAtTime((currentPosition) * 1000, MediaMetadataRetriever.OPTION_CLOSEST);
            int width = bmFrame.getWidth();
            int height = bmFrame.getHeight();

            int size = bmFrame.getRowBytes() * bmFrame.getHeight();
            ByteBuffer byteBuffer = ByteBuffer.allocate(size);
            bmFrame.copyPixelsToBuffer(byteBuffer);
            byte[] byteArray = byteBuffer.array();
            tracker.onFrame(
                    previewWidth,
                    previewHeight,
                    getLuminanceStride(),
                    0,
                    byteArray,
                    timestamp);
            trackingOverlay.postInvalidate();
            Log.i( "hank_debug", "pos:" + currentPosition);
            if (bmFrame == null) {
                Toast.makeText(VideoDetectorActivity.this, "bmFrame == null!", Toast.LENGTH_LONG).show();
                mHandlerTime.postDelayed(this, 100);
                computingDetection = false;
            } else {
                //val myCaptureDialog = android.app.AlertDialog.Builder(this@VideoActivity)
                //val capturedImageView = ImageView(this@VideoActivity)
                //capturedImageView.setImageBitmap(bmFrame)

                try {

                    bmFrame = Bitmap.createScaledBitmap(bmFrame, TF_OD_API_INPUT_SIZE, TF_OD_API_INPUT_SIZE, false);
                    if(0==1)
                    {//show image for debug
                        AlertDialog.Builder myCaptureDialog =
                                new AlertDialog.Builder(VideoDetectorActivity.this);
                        ImageView capturedImageView = new ImageView(VideoDetectorActivity.this);
                        capturedImageView.setImageBitmap(bmFrame);
                        LayoutParams capturedImageViewLayoutParams =
                                new LayoutParams(LayoutParams.WRAP_CONTENT,
                                        LayoutParams.WRAP_CONTENT);
                        capturedImageView.setLayoutParams(capturedImageViewLayoutParams);

                        myCaptureDialog.setView(capturedImageView);
                        AlertDialog dialog = myCaptureDialog.create();
                        dialog.requestWindowFeature(Window.FEATURE_NO_TITLE);
                        WindowManager.LayoutParams wmlp = dialog.getWindow().getAttributes();

                        wmlp.gravity = Gravity.TOP | Gravity.LEFT;
                        wmlp.x = 100;
                        wmlp.y = 300;
                        dialog.show();
                    }
                    Bitmap finalBmFrame = bmFrame;
                    runInBackground(
                            new Runnable() {
                                @Override
                                public void run() {
                                    final long startTime = SystemClock.uptimeMillis();
                                    final List<Classifier.Recognition> results = detector.recognizeImage(finalBmFrame);
                                    lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;

                                    cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
                                    final Canvas canvas = new Canvas(cropCopyBitmap);
                                    final Paint paint = new Paint();
                                    paint.setColor(Color.RED);
                                    paint.setStyle(Style.STROKE);
                                    paint.setStrokeWidth(2.0f);


                                    final List<Classifier.Recognition> mappedRecognitions =
                                            new LinkedList<Classifier.Recognition>();

                                    for (final Classifier.Recognition result : results) {
                                        final RectF location = result.getLocation();
                                        //Log.i("hank_confidence:", result.getConfidence().toString());
                                        if (location != null && result.getConfidence() >= MINIMUM_CONFIDENCE_TF_OD_API) {
                                            //adjust the rectangle location
                                            float tmp = location.top;
                                            float shift = 17;
                                            location.top = location.top>shift?location.top-shift:0;
                                            location.bottom = location.bottom>shift?location.bottom-shift:location.bottom-tmp;
                                            Log.i( "hank_debug", "b:" + location.bottom + " t:" + location.top);
                                            canvas.drawRect(location, paint);

                                            cropToFrameTransform.mapRect(location);
                                            result.setLocation(location);
                                            mappedRecognitions.add(result);
                                        }
                                    }

                                    tracker.trackResults(mappedRecognitions, luminanceCopy, currentPosition);
                                    trackingOverlay.postInvalidate();

                                    requestRender();
                                    //mHandlerTime.postDelayed(this, regularPollingTime);
                                    computingDetection = false;
                                }
                            }
                    );

                } catch (Exception e) {
                    e.printStackTrace();
                }
                mHandlerTime.postDelayed(this, regularPollingTime);
                //val capturedImageViewLayoutParams = android.app.ActionBar.LayoutParams(android.app.ActionBar.LayoutParams.WRAP_CONTENT, android.app.ActionBar.LayoutParams.WRAP_CONTENT)
                //capturedImageView.setLayoutParams(capturedImageViewLayoutParams)

                //myCaptureDialog.setView(capturedImageView)
                //myCaptureDialog.show()
                //mHandlerTime.postDelayed(this, 1000)
            }

            //txtVideoResult.setText(currentPosition.toString())


        }

    };

    public final void chooseVideoFromGallary() {
        Intent galleryIntent = new Intent("android.intent.action.PICK", MediaStore.Video.Media.EXTERNAL_CONTENT_URI);
        startActivityForResult(Intent.createChooser(galleryIntent, "Select File"), this.CHOOSE_VIDEO);
    }

    private final void takeVideoFromCamera() {
        Intent intent = new Intent("android.media.action.VIDEO_CAPTURE");
        this.startActivityForResult(intent, this.CHOOSE_CAMERA);
    }

    private final void showPictureDialog() {
        android.app.AlertDialog.Builder pictureDialog = new android.app.AlertDialog.Builder((Context)this);
        pictureDialog.setTitle((CharSequence)"Select Action");
        String[] pictureDialogItems = new String[]{"Select video from gallery", "Record video from camera"};
        pictureDialog.setItems((CharSequence[])pictureDialogItems, (android.content.DialogInterface.OnClickListener)(new android.content.DialogInterface.OnClickListener() {
            @Override
            public final void onClick(DialogInterface dialog, int which) {
                switch(which) {
                    case 0:
                        VideoDetectorActivity.this.chooseVideoFromGallary();
                        break;
                    case 1:
                        VideoDetectorActivity.this.takeVideoFromCamera();
                        break;
                    default:
                        break;
                }

            }
        }));
        pictureDialog.show();
    }

    @Override
    protected void onCreate(final Bundle savedInstanceState) {
        LOGGER.d("onCreate " + this);
        setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE);
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_video);
        mediaMetadataRetriever = new MediaMetadataRetriever();
        //getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        //setContentView(R.layout.activity_camera);

        if (hasPermission()) {

            //set the layout funtion
            videoView = findViewById(R.id.vv);
            Log.i( TAG, "vv height:" + videoView.getLayoutParams().height + " width:" + videoView.getLayoutParams().width);
            ViewTreeObserver vto = videoView.getViewTreeObserver();

            vto.addOnPreDrawListener(new ViewTreeObserver.OnPreDrawListener() {
                @Override
                public boolean onPreDraw() {
                    videoView.getViewTreeObserver().removeOnPreDrawListener(this);
                    int finalHeight = videoView.getLayoutParams().height;
                    int finalWidth = videoView.getLayoutParams().width;
                    Log.d(TAG , "" + "Height: " + finalHeight + " Width: " + finalWidth);
                    return true;
                }
            });

            btnCamera = (Button) findViewById(R.id.btnCamera);
            btnCamera.setOnClickListener((View it) ->
                    VideoDetectorActivity.this.chooseVideoFromGallary());
            //setup();
        } else {
            requestPermission();
        }
    }

    public void setupTracker() {
        final float textSizePx =
                TypedValue.applyDimension(
                        TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
        borderedText = new BorderedText(textSizePx);
        borderedText.setTypeface(Typeface.MONOSPACE);

        tracker = new MultiBoxTracker(this);

        int cropSize = TF_OD_API_INPUT_SIZE;

        try {
            detector =
                    TFLiteObjectDetectionAPIModel.create(
                            getAssets(),
                            TF_OD_API_MODEL_FILE,
                            TF_OD_API_LABELS_FILE,
                            TF_OD_API_INPUT_SIZE,
                            TF_OD_API_IS_QUANTIZED);
            cropSize = TF_OD_API_INPUT_SIZE;
        } catch (final IOException e) {
            LOGGER.e("Exception initializing classifier!", e);
            Toast toast =
                    Toast.makeText(
                            getApplicationContext(), "Classifier could not be initialized", Toast.LENGTH_SHORT);
            toast.show();
            finish();
        }



        //sensorOrientation = rotation - getScreenOrientation();
        //LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

        LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
        rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);
        croppedBitmap = Bitmap.createBitmap(cropSize, cropSize, Config.ARGB_8888);

        frameToCropTransform =
                ImageUtils.getTransformationMatrix(
                        previewWidth, previewHeight,
                        cropSize, cropSize,
                        0, MAINTAIN_ASPECT);

        cropToFrameTransform = new Matrix();
        frameToCropTransform.invert(cropToFrameTransform);

        trackingOverlay = (OverlayView) findViewById(R.id.tracking_overlay);
        trackingOverlay.bringToFront();
        trackingOverlay.addCallback(
                new DrawCallback() {
                    @Override
                    public void drawCallback(final Canvas canvas) {
                        Log.i( "hank_debug", "draw overlay");
                        tracker.draw(canvas);
                        if (isDebug()) {
                            tracker.drawDebug(canvas);
                        }
                    }
                });

        addCallback(
                new DrawCallback() {
                    @Override
                    public void drawCallback(final Canvas canvas) {
                        if (!isDebug()) {
                            return;
                        }
                        final Bitmap copy = cropCopyBitmap;
                        if (copy == null) {
                            return;
                        }

                        final int backgroundColor = Color.argb(100, 0, 0, 0);
                        canvas.drawColor(backgroundColor);

                        final Matrix matrix = new Matrix();
                        final float scaleFactor = 2;
                        matrix.postScale(scaleFactor, scaleFactor);
                        matrix.postTranslate(
                                canvas.getWidth() - copy.getWidth() * scaleFactor,
                                canvas.getHeight() - copy.getHeight() * scaleFactor);
                        canvas.drawBitmap(copy, matrix, new Paint());

                        final Vector<String> lines = new Vector<String>();
                        if (detector != null) {
                            final String statString = detector.getStatString();
                            final String[] statLines = statString.split("\n");
                            for (final String line : statLines) {
                                lines.add(line);
                            }
                        }
                        lines.add("");

                        lines.add("Frame: " + previewWidth + "x" + previewHeight);
                        lines.add("Crop: " + copy.getWidth() + "x" + copy.getHeight());
                        lines.add("View: " + canvas.getWidth() + "x" + canvas.getHeight());
                        lines.add("Rotation: " + sensorOrientation);
                        lines.add("Inference time: " + lastProcessingTimeMs + "ms");

                        borderedText.drawLines(canvas, 10, canvas.getHeight() - 10, lines);
                    }
                });
    }

    public void addCallback(final OverlayView.DrawCallback callback) {
        final OverlayView overlay = (OverlayView) findViewById(R.id.debug_overlay);
        if (overlay != null) {
            overlay.addCallback(callback);
            overlay.bringToFront();
        }
    }

    protected int[] getRgbBytes() {
        imageConverter.run();
        return rgbBytes;
    }

    protected int getLuminanceStride() {
        return yRowStride;
    }

    protected byte[] getLuminance() {
        return yuvBytes[0];
    }

    public boolean isDebug() {
        return debug;
    }

    protected synchronized void runInBackground(final Runnable r) {
        if (handler != null) {
            handler.post(r);
        }
    }

    protected void readyForNextImage() {
        if (postInferenceCallback != null) {
            postInferenceCallback.run();
        }
    }

    @Override
    public synchronized void onStart() {
        LOGGER.d("onStart " + this);
        super.onStart();
    }

    @Override
    public synchronized void onResume() {
        LOGGER.d("onResume " + this);
        super.onResume();

        handlerThread = new HandlerThread("inference");
        handlerThread.start();
        handler = new Handler(handlerThread.getLooper());
    }


    @Override
    public synchronized void onStop() {
        LOGGER.d("onStop " + this);
        super.onStop();
    }

    @Override
    public synchronized void onDestroy() {
        LOGGER.d("onDestroy " + this);
        super.onDestroy();
    }


    public void requestRender() {
        final OverlayView overlay = (OverlayView) findViewById(R.id.debug_overlay);
        if (overlay != null) {
            overlay.postInvalidate();
        }
    }


    OverlayView trackingOverlay;


}
