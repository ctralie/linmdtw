/**
 * A parallel implementation of DTW
 * 
 * @param {*} xpoints 
 * @param {*} ypoints 
 * @param {*} isSubSequence 
 * 
 * @returns {float} Cost (Start by just computing the cost,     
 *                      and compare to lower right element in accumulated cost for full DTW)
 */
function matchPointsParallel(xpoints, ypoints, isSubSequence) {

}

function matchPoints(xpoints, ypoints, isSubSequence){
    var xValue = 0;
    var yValue = 1;

    //creating a 2d array to hold the matching and distances
    var distances = [];
    var matches = [];	
    var accumulated_dists = [];

    for (var i = 0; i < xpoints.length; i++){
        distances.push([]);
        accumulated_dists.push([]);
        
        for (var j = 0; j < ypoints.length; j++){

            distances[i][j] = computeDistance(xpoints[i][xValue], xpoints[i][yValue], 
            ypoints[j][xValue], ypoints[j][yValue]);
            
            accumulated_dists[i].push(0);
        }
    } 	

    if(!isSubSequence){

        //the very first element
        accumulated_dists[0][0] = (distances[0][0]);

        //fill first row
        for(var i = 1; i <= ypoints.length -1; i++){
            accumulated_dists[0][i] = (distances[0][i] + accumulated_dists[0][i-1]);
        }

        //fill first col
        for(var i = 1; i <= xpoints.length -1; i++){
            accumulated_dists[i][0] = (distances[i][0] + accumulated_dists[i-1][0]);
        }

        //fill the rest of the accumulated cost array
        for(var i = 1; i <= xpoints.length -1; i++){
            for(var j = 1; j <= ypoints.length -1; j++){
            
                //the total distance will come from the minimum of the three options added to the distance
                accumulated_dists[i][j] = (Math.min(accumulated_dists[i-1][j-1], accumulated_dists[i-1][j], 
                accumulated_dists[i][j-1]) + distances[i][j]);
            
            }
        }

        //starting at the end of the table
        var i = xpoints.length - 1;
        var j = ypoints.length - 1;
    
        //backtracing!
    
        //making sure the last two point sets match
        matches.push([xpoints.length-1,ypoints.length-1]);

        do{
            //if either i or j are zero, we know we can only go back in one direction from there
            if(i == 0){
            j--;
            }else if (j == 0){
            i --;
            }else{
                //if the next row over is the min, go to the next row over
                if(accumulated_dists[i-1][j] == Math.min(accumulated_dists[i-1][j-1], accumulated_dists[i-1][j], accumulated_dists[i][j-1])){
                    i--;
                //if the next column over is the min, go to the next col
                }else if (accumulated_dists[i][j-1] == Math.min(accumulated_dists[i-1][j-1], accumulated_dists[i-1][j], accumulated_dists[i][j-1])){
                    j--;
                //if neither of those are true, then it must be the diagonal, so we decrease i and j
                }else{
                    i--;
                    j--;
                }
            }
            matches.push([i,j]);
       }while(i>0 || j>0)

    }else if (isSubSequence){
        var subDists = [[distances[0][0]]];

        //fill first row
        for (let i = 1; i <= ypoints.length - 1; i++){
            subDists[0][i] = distances[0][i];
        }

        //fill first col
        for(let i = 1; i <= xpoints.length -1; i++){
            subDists.push([]);
            subDists[i][0] = distances[i][0] + subDists[i-1][0];
        }

        //make the sub sequence specific "accumulated distances" array
        for(let i = 1; i <= xpoints.length -1; i++){
            for(let j = 1; j <= ypoints.length -1; j++){
                //the total distance will come from the minimum of the three options added to the distance
                subDists[i][j] = Math.min(subDists[i-1][j-1], subDists[i-1][j], 
                                            subDists[i][j-1]) + distances[i][j];
            }
        }
        matches = subSequence(subDists, matches);
    }

    //return the match list instead of calling the drawing
    return matches;
}

function computeDistance(x1, y1, x2, y2){
    //calculate the distance between the two points using euclidean distance
    //||d|| = √( (x1 – y1)^2 + (x2 - y2)^2 )
    var distance = Math.sqrt( Math.pow( (x1 - x2), 2) + Math.pow( (y1 - y2), 2) );

    return distance;
}

function subSequence(subDists, matches){
    //finding the minimum element at the bottom row
    //variable to hold the minimum value
    let minimumVal = subDists[xpoints.length-1][0];
    let minimum = [xpoints.length-1, 0];

    //maybe put xpoint.length - 1 into a variable so you don't keep referencing it and looking it up

    //finding the minimum value on the bottom row

    for(let j = 1; j < ypoints.length; j++){
        //if the current distance is less than the minimum so far
        if(subDists[xpoints.length-1][j] < minimumVal){
            //minimum is found and is stored in a variable
            minimumVal = subDists[xpoints.length-1][j];
            minimum = [xpoints.length - 1, j];
        }
    }

    //backtracing to the top row
    matches.push(minimum);  //should be the last element
    let i = minimum[0];
    let j = minimum[1];

    do{
        //if either i or j are zero, we know we can only go back in one direction from there
        if (j == 0){
            i --;
        }else{
            //if the next row over is the min, go to the next row over
            if(subDists[i-1][j] == Math.min(subDists[i-1][j-1], subDists[i-1][j], subDists[i][j-1])){
                i--;
            //if the next column over is the min, go to the next col
            }else if (subDists[i][j-1] == Math.min(subDists[i-1][j-1], subDists[i-1][j], subDists[i][j-1])){
                j--;
            //if neither of those are true, then it must be the diagonal, so we decrease i and j
            }else{
                i--;
                j--;
            }
        }
        matches.push([i,j]);
    //keep looping until we reach the top row, i.e. i == 0
    }while(i>0)

    return matches;
}