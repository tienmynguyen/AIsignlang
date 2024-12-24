package com.google.mediapipe.examples.handlandmarker

import android.content.Context
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.channels.FileChannel

class KeyPointClassifier(private val context: Context) {

    private var interpreter: Interpreter? = null

    init {
        try {
            val options = Interpreter.Options().apply {
                setNumThreads(1)
            }
            interpreter = Interpreter(loadModelFile(context), options)
            interpreter?.allocateTensors()
        } catch (e: Exception) {
            Log.e("KeyPointClassifier", "Error initializing TensorFlow Lite interpreter", e)
        }
    }

    fun classify(landmarkList: List<Float>): Int {
        return try {
            // Kiểm tra kích thước của landmarkList
            if (landmarkList.size != 42) {
                Log.e("KeyPointClassifier", "Invalid landmark list size: ${landmarkList.size}")
                return -1 // Hoặc xử lý lỗi theo cách khác
            }

            // Chuyển đổi landmarkList sang mảng 3D [1, 21, 2]
            val inputArray = landmarkList.toFloatArray().reshape(1, 21, 2)

            // Tạo input buffer và ghi dữ liệu vào
            val inputBuffer = convertArrayToByteBuffer(inputArray)

            // Tạo output buffer với kích thước 27 * 4 = 108 bytes (FLOAT32)
            val outputBuffer = ByteBuffer.allocateDirect(27 * 4)
            outputBuffer.order(java.nio.ByteOrder.nativeOrder())

            // Chạy inference
            interpreter?.run(inputBuffer, outputBuffer)

            // Lấy kết quả từ output buffer (FLOAT32)
            outputBuffer.rewind()
            val outputArray = FloatArray(27) // 27 phần tử
            outputBuffer.asFloatBuffer().get(outputArray)

            // Tìm index của giá trị lớn nhất trong outputArray
            var maxIndex = 0
            var maxValue = outputArray[0]
            for (i in 1 until outputArray.size) {
                if (outputArray[i] > maxValue) {
                    maxValue = outputArray[i]
                    maxIndex = i
                }
            }
            maxIndex
        } catch (e: Exception) {
            Log.e("KeyPointClassifier", "Error classifying landmark list", e)
            -1 // Hoặc xử lý lỗi theo cách khác phù hợp với ứng dụng của bạn
        }
    }

    private fun loadModelFile(context: Context): ByteBuffer {
        val assetFileDescriptor = context.assets.openFd("keypoint_classifier.tflite") // Đường dẫn đến model trong assets
        val inputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = assetFileDescriptor.startOffset
        val declaredLength = assetFileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    private fun convertArrayToByteBuffer(array: Array<Array<FloatArray>>): ByteBuffer {
        val buffer = ByteBuffer.allocateDirect(array.size * array[0].size * array[0][0].size * 4) // 4 bytes per float
        buffer.order(java.nio.ByteOrder.nativeOrder())
        for (dim1 in array) {
            for (dim2 in dim1) {
                for (value in dim2) {
                    buffer.putFloat(value)
                }
            }
        }
        buffer.rewind()
        return buffer
    }

    private fun FloatArray.reshape(dim1: Int, dim2: Int, dim3: Int): Array<Array<FloatArray>> {
        val result = Array(dim1) { Array(dim2) { FloatArray(dim3) } }
        var index = 0
        for (i in 0 until dim1) {
            for (j in 0 until dim2) {
                for (k in 0 until dim3) {
                    result[i][j][k] = this[index++]
                }
            }
        }
        return result
    }
}