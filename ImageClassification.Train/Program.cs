﻿using ImageClassification.Shared.DataModels;
using ImageClassification.Shared.Utils;
using ImageClassification.Train.Common;
using ImageClassification.Train.DataModels;
using Microsoft.ML;
using Microsoft.ML.Vision;
using System.Diagnostics;
using static Microsoft.ML.Transforms.ValueToKeyMappingEstimator;

namespace ImageClassification.Train
{
    internal class Program
    {
        static void Main(string[] args)
        {
            const string assetsRelativePath = @"../../../assets";
            string assetsPath = GetAbsolutePath(assetsRelativePath);

            string outputMlNetModelFilePath = Path.Combine(assetsPath, "outputs", "classifier.zip");
            string imagesFolderPathForPredictions = Path.Combine(assetsPath, "inputs", "test-images");

            string fullImagesetFolderPath = Path.Combine(assetsPath, "inputs", "images", "dataset");

            MLContext mlContext = new MLContext(seed: 1);

            // Specify MLContext Filter to only show feedback log/traces about ImageClassification
            // This is not needed for feedback output if using the explicit MetricsCallback parameter
            mlContext.Log += FilterMLContextLog;

            // 2. Load the initial full image-set into an IDataView and shuffle so it'll be better balanced
            IEnumerable<ImageData> images = LoadImagesFromDirectory(folder: fullImagesetFolderPath, useFolderNameAsLabel: true);
            IDataView fullImagesDataset = mlContext.Data.LoadFromEnumerable(images);
            IDataView shuffledFullImageFilePathsDataset = mlContext.Data.ShuffleRows(fullImagesDataset);

            // 3. Load Images with in-memory type within the IDataView and Transform Labels to Keys (Categorical)
            IDataView shuffledFullImagesDataset = mlContext.Transforms.Conversion
                .MapValueToKey(outputColumnName: "LabelAsKey", inputColumnName: "Label", keyOrdinality: KeyOrdinality.ByValue)
                .Append(mlContext.Transforms.LoadRawImageBytes(
                                                outputColumnName: "Image",
                                                imageFolder: fullImagesetFolderPath,
                                                inputColumnName: "ImagePath"))
                .Fit(shuffledFullImageFilePathsDataset)
                .Transform(shuffledFullImageFilePathsDataset);

            // 4. Split the data 80:20 into train and test sets, train and evaluate.
            DataOperationsCatalog.TrainTestData trainTestData = mlContext.Data.TrainTestSplit(shuffledFullImagesDataset, testFraction: 0.2);
            IDataView trainDataView = trainTestData.TrainSet;
            IDataView testDataView = trainTestData.TestSet;

            // 5. Define the model's training pipeline using DNN default values
            //Microsoft.ML.Data.EstimatorChain<Microsoft.ML.Transforms.KeyToValueMappingTransformer> pipeline = mlContext.MulticlassClassification.Trainers.SdcaNonCalibrated(featureColumnName: "Image",
            //                             labelColumnName: "LabelAsKey"
            //                             /*validationSet: testDataView*/)
            //    .Append(mlContext.Transforms.Conversion.MapKeyToValue(outputColumnName: "PredictedLabel",
            //                                                          inputColumnName: "PredictedLabel"));

            // 5.1 (OPTIONAL) Define the model's training pipeline by using explicit hyper-parameters
            ImageClassificationTrainer.Options options = new ImageClassificationTrainer.Options()
            {
                FeatureColumnName = "Image",
                LabelColumnName = "LabelAsKey",
                // Just by changing/selecting InceptionV3/MobilenetV2/ResnetV250  
                // you can try a different DNN architecture (TensorFlow pre-trained model). 
                Arch = ImageClassificationTrainer.Architecture.MobilenetV2,
                Epoch = 1000,       //100
                BatchSize = 10,
                LearningRate = 0.01f,
                MetricsCallback = (metrics) => Console.WriteLine(metrics),
                ValidationSet = testDataView
            };

            Microsoft.ML.Data.EstimatorChain<Microsoft.ML.Transforms.KeyToValueMappingTransformer> pipeline = mlContext.MulticlassClassification.Trainers.ImageClassification(options)
                    .Append(mlContext.Transforms.Conversion.MapKeyToValue(
                        outputColumnName: "PredictedLabel",
                        inputColumnName: "PredictedLabel"));

            // 6. Train/create the ML model
            Console.WriteLine("*** Training the image classification model with DNN Transfer Learning on top of the selected pre-trained model/architecture ***");

            // Measuring training time
            Stopwatch watch = Stopwatch.StartNew();

            //Train
            ITransformer trainedModel = pipeline.Fit(trainDataView);

            watch.Stop();
            long elapsedMs = watch.ElapsedMilliseconds;

            Console.WriteLine($"Training with transfer learning took: {elapsedMs / 1000} seconds");

            // 7. Get the quality metrics (accuracy, etc.)
            EvaluateModel(mlContext, testDataView, trainedModel);

            // 8. Save the model to assets/outputs (You get ML.NET .zip model file and TensorFlow .pb model file)
            mlContext.Model.Save(trainedModel, trainDataView.Schema, outputMlNetModelFilePath);
            Console.WriteLine($"Model saved to: {outputMlNetModelFilePath}");

            // 9. Try a single prediction simulating an end-user app
            TrySinglePrediction(imagesFolderPathForPredictions, mlContext, trainedModel);

            Console.WriteLine("Press any key to finish");
            Console.ReadKey();
        }

        public static string GetAbsolutePath(string relativePath)
            => FileUtils.GetAbsolutePath(typeof(Program).Assembly, relativePath);

        public static IEnumerable<ImageData> LoadImagesFromDirectory(
            string folder,
            bool useFolderNameAsLabel = true)
            => FileUtils.LoadImagesFromDirectory(folder, useFolderNameAsLabel)
                .Select(x => new ImageData(x.imagePath, x.label));

        private static void EvaluateModel(MLContext mlContext, IDataView testDataset, ITransformer trainedModel)
        {
            Console.WriteLine("Making predictions in bulk for evaluating model's quality...");

            // Measuring time
            Stopwatch watch = Stopwatch.StartNew();

            IDataView predictionsDataView = trainedModel.Transform(testDataset);

            Microsoft.ML.Data.MulticlassClassificationMetrics metrics = mlContext.MulticlassClassification.Evaluate(predictionsDataView, labelColumnName: "LabelAsKey", predictedLabelColumnName: "PredictedLabel");
            ConsoleHelper.PrintMultiClassClassificationMetrics("TensorFlow DNN Transfer Learning", metrics);

            watch.Stop();
            long elapsed2Ms = watch.ElapsedMilliseconds;

            Console.WriteLine($"Predicting and Evaluation took: {elapsed2Ms / 1000} seconds");
        }

        private static void TrySinglePrediction(string imagesFolderPathForPredictions, MLContext mlContext, ITransformer trainedModel)
        {
            // Create prediction function to try one prediction
            PredictionEngine<InMemoryImageData, ImagePrediction> predictionEngine = mlContext.Model
                .CreatePredictionEngine<InMemoryImageData, ImagePrediction>(trainedModel);

            IEnumerable<InMemoryImageData> testImages = FileUtils.LoadInMemoryImagesFromDirectory(
                imagesFolderPathForPredictions, false);

            foreach (InMemoryImageData testImage in testImages)
            {
                InMemoryImageData imageToPredict = testImage;

                ImagePrediction prediction = predictionEngine.Predict(imageToPredict);

                Console.WriteLine(
                    $"Image Filename : [{imageToPredict.ImageFileName}], " +
                    $"Scores : [{string.Join(",", prediction.Score)}], " +
                    $"Predicted Label : {prediction.PredictedLabel}");
            }
        }

        private static void FilterMLContextLog(object sender, LoggingEventArgs e)
        {
            if (e.Message.StartsWith("[Source=ImageClassificationTrainer;"))
            {
                Console.WriteLine(e.Message);
            }
        }
    }
}