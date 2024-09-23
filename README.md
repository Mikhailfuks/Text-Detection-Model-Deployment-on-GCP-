using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;

public class RealEstatePricePrediction
{
    public class RealEstateData
    {
        [LoadColumn(0)] public float SquareFeet;
        [LoadColumn(1)] public int Bedrooms;
        [LoadColumn(2)] public int Bathrooms;
        [LoadColumn(3)] public float Price; // Target variable
    }

    public static void Main(string[] args)
    {
        // Load data
        var mlContext = new MLContext();
        var data = mlContext.Data.LoadFromTextFile<RealEstateData>("real_estate_data.csv", hasHeader: true, separator: ',');

        // Define the pipeline
        var pipeline = mlContext.Transforms
            .Concatenate("Features", "SquareFeet", "Bedrooms", "Bathrooms")
            .Append(mlContext.Regression.Trainers.Sdca(labelColumnName: "Price", featureColumnName: "Features"));

        // Train the model
        var model = pipeline.Fit(data);

        // Predict the price of a new property
        var newProperty = new RealEstateData { SquareFeet = 2000, Bedrooms = 3, Bathrooms = 2 };
        var pricePrediction = model.Transform(mlContext.Data.Create(newProperty));

        // Display the predicted price
        var prediction = pricePrediction.First();
        Console.WriteLine($"Predicted price: {prediction.Price}");
    }
}
