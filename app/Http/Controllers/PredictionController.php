<?php

namespace App\Http\Controllers;

use App\models\Prediction;
use Illuminate\Http\Request;

class PredictionController extends Controller
{
    public function prediction(Request $request){
        $newPrediction = new Prediction;
        $newPrediction->lat = $request->latitude;
        $newPrediction->long = $request->longitude;
        $newPrediction->save();
        return view("prediction");
    }
}
