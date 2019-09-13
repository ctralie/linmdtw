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
*                       Then costs[ab] is the cost of matching a to b
                        costs[a] is the cost of deleting a
 * @return {number} Maximum possible score of matching x to y with a sequence of operation
                    (TODO: Also return an optimal sequence of operations to match x to y)
 */
function matchStrings(x, y, alphabet, costs) {
    // TODO: Perform the matching.  As part of the output, write
    // out the dynamic programming matrix to a table

    //setting the length of the two strings in order to allocate memory
    var mLength = x.length;
    var nLength = y.length;

    //allocating memory in a 2 dimensional array
    for (var i = 0; i < mLength; i++){
        D.push([]);
        for (var j = 0; j < nLength; j++){
            D[i].push(0);
        }
    }    

    //----------------------Filling in the Table-----------------------
    D[0][0] = 0; //first element will always be 0

    //filling in the first row
    for (var i = 0; i < mLength; i++){ //going through each item in row 1
        var cost = 0; //temporary variable to hold the cost of each element
        for(var j = 0; j < alphabet.length; j++){ //going through each element of the alphabet
            if(x[i] == alphabet[j]){ //if the element in the first string == the current letter in the alphabet provided
                cost = costs.alphabet[j]; //set the cost equal to where the deletion for that letter is in costs[]
            }
        }
        D[0][i+1] = D[0][i] + cost; //the current cell's value = the cell to the left + the cost of the deletion
    }

    //filling in the first column
    for (var i = 0; i < nLength; i++){ //going through each element in col 1
        var cost = 0; //temporary variable to hold the cost of each item in col 1
        for(var j = 0; j < alphabet.length; j++){ //going through each element of the alphabet for matching
            if(y[i] == alphabet[j]){ //if the current letter in the second string == the current letter in the alphabet provided
                cost = costs.alphabet[j]; //the cost of deleting that char from the string from the given costs
            }
        }
        D[i+1][0] = D[i][0] + cost; //the current cell's value = the cell above + the cost of the deletion
    }

    //beatrice456734paul jack dadmommy  458965054589981 - helpful note from the kids I babysit
    

    // TODO: Fill in the dynamic programming array properly


    outputTable(x, y, D);
    // TODO: Return the correct results
    return 0.0;
}