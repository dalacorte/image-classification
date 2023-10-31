using ImageClassification.Api.Helper;
using ImageClassification.Api.ML.DataModels;
using ImageClassification.Shared.DataModels;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.ML;

namespace ImageClassification.Api.Controllers
{
    [ApiController]
    [Route("[controller]")]
    public class ImageController : ControllerBase
    {
        private readonly PredictionEnginePool<InMemoryImageData, ImagePrediction> _predictionEnginePool;

        public ImageController(PredictionEnginePool<InMemoryImageData, ImagePrediction> predictionEnginePool)
        {
            _predictionEnginePool = predictionEnginePool;
        }

        [HttpPost]
        [ProducesResponseType(200)]
        [ProducesResponseType(400)]
        [Route("classify")]
        public async Task<IActionResult> Classify(IFormFile imageFile)
        {
            if (imageFile.Length == 0)
                return BadRequest();

            MemoryStream imageMemoryStream = new MemoryStream();
            await imageFile.CopyToAsync(imageMemoryStream);

            // Check that the image is valid.
            byte[] imageData = imageMemoryStream.ToArray();
            if (!imageData.IsValidImage())
                return StatusCode(StatusCodes.Status415UnsupportedMediaType);

            // Measure execution time.
            System.Diagnostics.Stopwatch watch = System.Diagnostics.Stopwatch.StartNew();

            // Set the specific image data into the ImageInputData type used in the DataView.
            InMemoryImageData imageInputData = new InMemoryImageData(image: imageData, label: null, imageFileName: null);

            // Predict code for provided image.
            ImagePrediction prediction = _predictionEnginePool.Predict(imageInputData);

            // Stop measuring time.
            watch.Stop();
            long elapsedMs = watch.ElapsedMilliseconds;

            // Predict the image's label (The one with highest probability).
            ImagePredictedLabelWithProbability imageBestLabelPrediction =
                new ImagePredictedLabelWithProbability
                {
                    PredictedLabel = prediction.PredictedLabel,
                    Probability = prediction.Score.Max(),
                    PredictionExecutionTime = elapsedMs,
                    ImageId = imageFile.FileName,
                };

            return Ok(imageBestLabelPrediction);
        }
    }
}