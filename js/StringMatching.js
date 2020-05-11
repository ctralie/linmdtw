/**
 * A function to output the 2D dynamic programming array associated to 
 * a string matching problem of a length M string to a length N string
 * to an HTML table.  Note that the dynamic programming matrix is (M+1)x(N+1)
 * but it actually outputs to an (M+2)x(N+2) table, since it shows the strings
 * next to the dynamic programming entries
 * 
 * @param {string} x First string (length M)
 * @param {string} y Second string (length N)
 * @param {2D array} D dynamic programming matrix (M+1 x N+1)
 */
 
function outputTable(x, y, D) {
   // Output the table
   let table = document.getElementById("mytable");
   table.innerHTML = ''; // Clear the table
   // Add a row for string 2 along the columns
   let row = document.createElement("tr");
   let col = null;
   for (var i = 0; i <= D[0].length; i++) {
       col = document.createElement("td");
       if (i == 1) {
           col.innerHTML = "∅";
       }
       else if (i > 1){
           col.innerHTML = y[i-2];
       }
       row.appendChild(col);
   }
   table.appendChild(row);
   //Now add all of the elements inside
   for (var i = 0; i < D.length; i++) {
       row = document.createElement("tr");
       table.appendChild(row);
       col = document.createElement("td");
       if (i == 0) {
           col.innerHTML = "∅";
       }
       else{
           col.innerHTML = x[i-1];
       }
       row.appendChild(col);
       for (var j = 0; j < D[i].length; j++) {
           col = document.createElement("td");
           row.appendChild(col);
           col.innerHTML = D[i][j];
       }
   }
}

/**
 * A function to compute a matching of two strings using the Needleman-Wunsch Algorithm
 * 
 * @param {string} x First string
 * @param {string} y Second string
 * @param {string} alphabet String consisting of alphabet (e.g. "ab" is only as and bs, where a is 0 and b is 1)
 * @param {dictionary} costs Costs of matching characters to each other.  
 *                      Let a be one type of character and b be another type of character.
 *                      Then costs[ab] is the cost of matching a to b
 *                      costs[a] is the cost of deleting a
 * @return {number} Maximum possible score of matching x to y with a sequence of operation
 * @return {string} The optimal path to match x and y
 */
function matchStrings(x, y, alphabet, costs) {
    //setting the length of the two strings in order to allocate memory
    var mLength = x.length;
    var nLength = y.length;

    let D = [];
    var P = []; //P for path, this array will keep track of the optimal pathway

    //allocating memory for the table and the path
    for (var i = 0; i < mLength + 1; i++){
        D.push([]);
        P.push([]);
        for (var j = 0; j < nLength + 1; j++){
            D[i].push(0);
            P[i].push(''); //for null values
        }
    }    

    //----------------------Filling in the Table-----------------------

    //filling in the first row
    for (var i = 0; i < nLength; i++){ //going through each item in row 1 (string y)
        var cost = 0; //temporary variable to hold the cost of each element

        for(var j = 0; j < alphabet.length; j++){ //going through each element of the alphabet item
            if(y[i] == alphabet[j]){ //if the element in y == the current letter in the alphabet provided
                
                //testing if there is a cost defined for this operation
                if(costs[alphabet[j]] == null){
                    console.log("The user has not defined the cost for deleting " + x[i]);
                }else{ //if the cost is defined, set it
                    cost = costs[alphabet[j]]; //set the cost equal to where the deletion for that letter is in costs[]
                }
            }
        }
        //D[row][column]
        D[0][i+1] = D[0][i] + cost; //the current cell's value = the cell to the left + the cost of the deletion
    }

    //filling in the first column
    for (var i = 0; i < mLength; i++){ //going through each element in col 1 (string x)
        var cost = 0; //temporary variable to hold the cost of each item in col 1

        for(var j = 0; j < alphabet.length; j++){ //going through each element of the alphabet for matching
            if(x[i] == alphabet[j]){ //if the current letter in the second string == the current letter in the alphabet provided
                
                //testing if there is a cost defined for this operation
                if(costs[alphabet[j]] == null){
                    console.log("The user has not defined the cost for deleting " + y[i]);
                }else{ //if the cost is defined, set it
                    cost = costs[alphabet[j]]; //the cost of deleting that char from the string from the given costs
                }
            }
        }

        //D[row][col]
        D[i+1][0] = D[i][0] + cost; //the current cell's value = the cell above + the cost of the deletion
    }

    //beatrice456734paul jack dadmommy  458965054589981 - helpful note from the kids I babysit
    
    //fill in the rest of the table

    //go through each row and column
    // x = row, mLength = x.length
    for (var i = 1; i <= mLength; i++){ //through the length of y - 1 because we already did the first element
        var diagCost = 0; //temp int to hold the cost for the diagonal cost
        var leftCost = 0; //temp to hold the left cost
        var upperCost = 0; //temp to hold cost sent from above

        //y = col, nLength = y.length
        for(var j = 1; j <= nLength; j++){//through the length of x - 1 ''
            //temp variables to hold the current correlating row and column
            var currentY = y[j-1]; //row
            var currentX = x[i-1]; //column

            //Testing if the current row letter has a deletion cost
            if(costs[currentY] == null){
                console.log("The user has not defined the cost for deleting " + currentX);
            }
            //now checking the column
            if (costs[currentX] == null){
                console.log("The user has not defined the cost for deleting " + currentY);
            }
            
            //--------------------Calculating Left, Upper, And Diagonal Costs-------------------

            //left cost will always be the deletion of the corresponding row
            //so we find what the header of that row is(currentX) and add the cost of 
            //deleting it to the cost to the left
            leftCost = D[i][j-1] + costs[currentY]; //j-1 because we go one column over
 
            //upper cost will always be the deletion of the corresponding column
            upperCost = D[i-1][j] + costs[currentX]; // i-1 because we go one row up

            //diagonal cost will be a little bit different, considering it's the match
            //of the current row and column header

            //there is a possibility that the row/col header could be switched, so we're
            //checking for both here
            if(!(costs[currentX + currentY] == null)){ //if x+y is not null
                diagCost = D[i-1][j-1] + costs[currentX + currentY];
            }else if(!(costs[currentY + currentX] == null)){ // if y+x is not null
                diagCost = D[i-1][j-1] + costs[currentY + currentX];
            }else{//if both are undefined, send error
                console.log("The user has not defined the cost for the matching of " + currentX + " and " + currentY);
            }

            //----------------------Finding The Optimal Path-------------------------
            //----------OPTIMAL DIAGONAL-------------
            if(diagCost > leftCost && diagCost > upperCost){ //diagonal must be the lowest of the three
                cost = diagCost;
                P[i][j] = 'd';
            //------------OPTIMAL LEFT-----------------
            }else if(leftCost > diagCost && leftCost > upperCost){
                cost = leftCost;
                P[i][j] = 'l';
            //------------OPTIMAL UPPER-----------------
            }else if(upperCost > diagCost && upperCost > leftCost){
                cost = upperCost;
                P[i][j] = 'u';
            //-----------Left and Upper the same----------------
            }else if(upperCost == leftCost){
                cost = leftCost; //just picking left
                P[i][j] = 'l'; 
            //-----------Diagonal and Left the same-------------
            }else if(diagCost == leftCost){
                cost = diagCost; //just picking diagonal
                P[i][j] = 'd';
            //-----------Diagonal and Upper the same----------
            }else if (diagCost == upperCost){
                cost = diagCost; //just picking diagonal
                P[i][j] = 'd';
            }

            //so we have determined the cost and the path, put the cost into the table
            D[i][j] = cost;
        }
    }

    outputTable(x, y, D);

    var backtrace_list  = computeBacktracing(P,x,y);

    return{
        score: D[nLength][mLength],
        backtrace: backtrace_list,
    };
}

/**
 * This function will use the path array to find the optimal path and store
 * in an array of strings to later be printed on the form.
 * 
 * @param {string} path         all paths
 * @param {string} x            the first string
 * @param {string} y            the second string
 * @return {string}  list_path, the list to be printed
 */
function computeBacktracing(path, x, y){
    var list_print = [];

    var currentX = "";
    var currentY = "";

    //starting at the veryyy end of the table
    var j = y.length; //just to keep it consistent with the code above, j is the cols
    var i = x.length;// and i is the rows

    //creating a variable to keep track of how many elements are in the printing list
    var n = 0;

    do{
        //holding the current string of the backtracing
        for(var l = 0; l < i; l++){
            currentX += x[l];
        }
        
        for(var m = 0; m < j; m++){
            currentY += y[m];
        }

        //just printing the two strings for right now, forgetting the cost
        list_print[n] = (currentX + " , " + currentY + ": ");

        if(path[i][j] == 'd'){
            //now go to the diagonal element
            i --;
            j --;
        }else if(path[i][j] == 'u'){
            //now go to the element above
            i --; //one row up
        }else if(path[i][j] == 'l'){
            //now fo to the element to the left
            j --; //one col over
        }

        n ++;
        currentX = "";
        currentY = "";

        console.log("i: " + i);
        console.log("j: " + j);

    }while (i > 0 || j > 0);

    list_print[list_print.length] = "0 , 0:";

    return list_print;
}
