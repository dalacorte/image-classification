using ImageClassification.Shared.DataModels;
using ImageClassification.Shared.Utils;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace ImageClassification.Predict
{
    internal class Program
    {
        private static void Main()
        {
            const string assetsRelativePath = @"../../../assets";
            string assetsPath = GetAbsolutePath(assetsRelativePath);

            string imagesFolderPathForPredictions = Path.Combine(assetsPath, "inputs", "images-for-predictions");

            string imageClassifierModelZipFilePath = Path.Combine(assetsPath, "inputs", "MLNETModel", "classifier.zip");

            try
            {
                MLContext mlContext = new MLContext(seed: 1);

                Console.WriteLine($"Loading model from: {imageClassifierModelZipFilePath}");

                // Load the model
                ITransformer loadedModel = mlContext.Model.Load(imageClassifierModelZipFilePath, out DataViewSchema? modelInputSchema);

                // Create prediction engine to try a single prediction (input = ImageData, output = ImagePrediction)
                PredictionEngine<InMemoryImageData, ImagePrediction> predictionEngine = mlContext.Model.CreatePredictionEngine<InMemoryImageData, ImagePrediction>(loadedModel);

                //Predict the first image in the folder
                IEnumerable<InMemoryImageData> imagesToPredict = FileUtils.LoadInMemoryImagesFromDirectory(imagesFolderPathForPredictions, false);

                InMemoryImageData imageToPredict = imagesToPredict.First();

                // Measure #1 prediction execution time.
                System.Diagnostics.Stopwatch watch = System.Diagnostics.Stopwatch.StartNew();

                ImagePrediction prediction = predictionEngine.Predict(imageToPredict);

                // Stop measuring time.
                watch.Stop();
                long elapsedMs = watch.ElapsedMilliseconds;
                Console.WriteLine("First Prediction took: " + elapsedMs + "mlSecs");

                // Measure #2 prediction execution time.
                System.Diagnostics.Stopwatch watch2 = System.Diagnostics.Stopwatch.StartNew();

                ImagePrediction prediction2 = predictionEngine.Predict(imageToPredict);

                // Stop measuring time.
                watch2.Stop();
                long elapsedMs2 = watch2.ElapsedMilliseconds;
                Console.WriteLine("Second Prediction took: " + elapsedMs2 + "mlSecs");

                // Get the highest score and its index
                float maxScore = prediction.Score.Max();

                ////////
                // Double-check using the index
                int maxIndex = prediction.Score.ToList().IndexOf(maxScore);
                VBuffer<ReadOnlyMemory<char>> keys = default;
                predictionEngine.OutputSchema[3].GetKeyValues(ref keys);
                ReadOnlyMemory<char>[] keysArray = keys.DenseValues().ToArray();
                ReadOnlyMemory<char> predictedLabelString = keysArray[maxIndex];
                ////////

                Console.WriteLine($"Image Filename : [{imageToPredict.ImageFileName}], " +
                                  $"Predicted Label : [{prediction.PredictedLabel}], " +
                                  $"Probability : [{maxScore}] "
                                  );

                //Predict all images in the folder
                //
                Console.WriteLine("");
                Console.WriteLine("Predicting several images...");

                foreach (InMemoryImageData currentImageToPredict in imagesToPredict)
                {
                    ImagePrediction currentPrediction = predictionEngine.Predict(currentImageToPredict);

                    Console.WriteLine(
                        $"Image Filename : [{currentImageToPredict.ImageFileName}], " +
                        $"Predicted Label : [{currentPrediction.PredictedLabel}], " +
                        $"Probability : [{currentPrediction.Score.Max()}]");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.ToString());
            }

            Console.WriteLine("Press any key to end the app..");
            Console.ReadKey();
        }

        public static string GetAbsolutePath(string relativePath)
            => FileUtils.GetAbsolutePath(typeof(Program).Assembly, relativePath);
    }
}