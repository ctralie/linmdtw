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

/**
 * This function will compute all of the matchings by creating the distance arrays and 
 * doing the backtracing of optimal matchings between points. Can be a sub-sequence or
 * normal DTW.
 * 
 * @param {num} xpoints     list of points on Line X
 * @param {num} ypoints     list of points on Line Y
 * @param {boolean} isSubSequence true if subsequence procedure, false if normal DTW
 * 
 * @returns matches -- the list of matched points
 */
function matchPoints(xpoints, ypoints, isSubSequence){
    let i = 0;
    let j = 0;
    var xValue = 0;
    var yValue = 1;

    //creating a 2d array to hold the matching and distances
    var distances = [];
    var matches = [];	
    var accumulated_dists = [];

    //allocating memory for accumulated_dists and populating the regular dists
    for (i = 0; i < xpoints.length; i++){
        distances.push([]);
        accumulated_dists.push([]);
        
        for (j = 0; j < ypoints.length; j++){

            //getting the euclidean distance for the table
            distances[i][j] = computeDistance(xpoints[i][xValue], xpoints[i][yValue], 
            ypoints[j][xValue], ypoints[j][yValue]);
            
            //allocating memory for accumulated distance table
            accumulated_dists[i].push(0);
        }
    } 	

    //----------------------FILLING ACCUMULATED DISTANCE TABLE---------------------

    //the very first element
    accumulated_dists[0][0] = (distances[0][0]);

    //if sub sequence, set the first row as straight dists, if not, accumulated
    if(isSubSequence){
        //fill first row as straight distances for sub sequence
        for (i = 1; i <= ypoints.length - 1; i++){
            accumulated_dists[0][i] = distances[0][i];
        }
    }else{
        //fill first row as accumulated for non sub sequence
        for(i = 1; i <= ypoints.length -1; i++){
            accumulated_dists[0][i] = (distances[0][i] + accumulated_dists[0][i-1]);
        }
    }

    //fill first col
    for(j = 1; j <= xpoints.length -1; j++){
        accumulated_dists[j][0] = (distances[j][0] + accumulated_dists[j-1][0]);
    }

    //fill the rest of the accumulated cost array
    for(i = 1; i <= xpoints.length -1; i++){
        for(j = 1; j <= ypoints.length -1; j++){
            
            //the total distance will come from the minimum of the three options added to the distance
            accumulated_dists[i][j] = (Math.min(accumulated_dists[i-1][j-1], accumulated_dists[i-1][j], 
                                                accumulated_dists[i][j-1]) + distances[i][j]);
            
        }
    }

    //-------------------------SETTING UP BACKTRACING----------------------------

    //setting the looping variables based on sub sequence or non
    if(isSubSequence){
        //subSequence starts at the minimum element on the bottom row, not necessarily the last
        let minimumVal = accumulated_dists[xpoints.length-1][0]; //holds the value
        let minimumCol = 0;            //holds the coordinates

        //finding the minimum
        for(let col = 1; col < ypoints.length; col++){
            //if the current distance is less than the minimum so far
            if(accumulated_dists[xpoints.length-1][col] < minimumVal){
                //minimum is found and is stored in a variable
                minimumVal = accumulated_dists[xpoints.length-1][col];
                minimumCol = col;
            }
        }

        //starts the match list at the minimum
        matches.push([xpoints.length-1, minimumCol]);

        //setting the loop boundaries
        i = xpoints.length - 1;
        j = minimumCol;
    }else{
        //starting at the end of the table
        i = xpoints.length - 1;
        j = ypoints.length - 1;

        //the last element in the table is where we start backtracing for the non-subSequence
        matches.push([xpoints.length-1,ypoints.length-1]);
    }

    //--------------------------------------BACKTRACING-----------------------------------------
        do{
            //if i == 0, and it is not a subsequence, continue left along the row
            if(i == 0 && !isSubSequence){
            j--;

            //if j == 0 continue up the column
            }else if (j == 0){
            i --;

            //i != 0  and j != 0
            }else{
                //if the next row over is the min, go to the next row over
                if(accumulated_dists[i-1][j] == Math.min(accumulated_dists[i-1][j-1], accumulated_dists[i-1][j], accumulated_dists[i][j-1])){
                    i--;

                //if the next column over is the min, go to the next col
                }else if (accumulated_dists[i][j-1] == Math.min(accumulated_dists[i-1][j-1], accumulated_dists[i-1][j],accumulated_dists[i][j-1])){
                    j--;

                //if neither of those are true, then it must be the diagonal, so we decrease i and j
                }else{
                    i--;
                    j--;
                }
            }

            //adding to the match list
            matches.push([i,j]);

        //while we're not at the top row, and for non-subSequence we are not at the first element
        }while(i>0 || (j>0 && !isSubSequence))    

    //return the match list instead of calling the drawing
    return matches;
}

/**
 * This function computes the euclidean distance between the two points passed in.
 * 
 * ex.
 *   point A = (x1,y1)
 *   point B = (x2,y2)
 * @param {num} x1 x-coordinate one
 * @param {num} y1 y-coordinate one
 * @param {num} x2 x-coordinate two
 * @param {num} y2 y-coordinate two 
 * 
 * @returns the euclidean distance 
 */
function computeDistance(x1, y1, x2, y2){
    //calculate the distance between the two points using euclidean distance
    //||d|| = √( (x1 – y1)^2 + (x2 - y2)^2 )
    var distance = Math.sqrt( Math.pow( (x1 - x2), 2) + Math.pow( (y1 - y2), 2) );

    return distance;
}