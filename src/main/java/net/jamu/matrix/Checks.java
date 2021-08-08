/*
 * Copyright 2019, 2021 Stefan Zobel
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package net.jamu.matrix;

import static net.jamu.matrix.DimensionsBase.checkCols;
import static net.jamu.matrix.DimensionsBase.checkRows;

final class Checks {

    static int checkArrayLength(int rows, int cols) {
        long length = (long) checkRows(rows) * (long) checkCols(cols);
        if (length > (long) Integer.MAX_VALUE) {
            throw new IllegalArgumentException(
                    "rows x cols (= " + length + ") exceeds the maximal possible length (= 2147483647) of an array");
        }
        return (int) length;
    }

    static int checkComplexArrayLength(int rows, int cols) {
        long length = 2L * ((long) checkRows(rows) * (long) checkCols(cols));
        if (length < 0L || length > (long) Integer.MAX_VALUE) {
            throw new IllegalArgumentException("rows x cols (= " + length
                    + ") exceeds the maximal possible length (= 2147483647) of a complex array");
        }
        return (int) length;
    }

    static void checkMult(Dimensions A, Dimensions B) {
        if (A.numColumns() != B.numRows()) {
            throw new IndexOutOfBoundsException(
                    "A.numColumns() != B.numRows() (" + A.numColumns() + " != " + B.numRows() + ")");
        }
    }

    static void checkTrans(Dimensions A, Dimensions AT) {
        if (A.numRows() != AT.numColumns()) {
            throw new IndexOutOfBoundsException(
                    "A.numRows() != AT.numColumns() (" + A.numRows() + " != " + AT.numColumns() + ")");
        }
        if (A.numColumns() != AT.numRows()) {
            throw new IndexOutOfBoundsException(
                    "A.numColumns() != AT.numRows() (" + A.numColumns() + " != " + AT.numRows() + ")");
        }
    }

    static void checkRequiredMinDimension(int rows, int cols, Dimensions X) {
        if (X.numRows() < rows) {
            throw new IndexOutOfBoundsException("X.numRows() < rows (" + X.numRows() + " < " + rows + ")");
        }
        if (X.numColumns() < cols) {
            throw new IndexOutOfBoundsException("X.numColumns() < cols (" + X.numColumns() + " < " + cols + ")");
        }
    }

    static void checkRequiredExactDimension(int rows, int cols, Dimensions X) {
        if (X.numRows() != rows) {
            throw new IndexOutOfBoundsException("X.numRows() != rows (" + X.numRows() + " != " + rows + ")");
        }
        if (X.numColumns() != cols) {
            throw new IndexOutOfBoundsException("X.numColumns() != cols (" + X.numColumns() + " != " + cols + ")");
        }
    }

    static void checkEqualDimension(Dimensions A, Dimensions B) {
        checkSameRows(A, B);
        checkSameCols(A, B);
    }

    static void checkCompatibleDimension(Dimensions A, int rows, int cols) {
        if (rows <= 0) {
            throw new IllegalArgumentException("rows must be strictly positive: " + rows);
        }
        if (cols <= 0) {
            throw new IllegalArgumentException("cols must be strictly positive: " + cols);
        }
        if (rows * cols != A.numRows() * A.numColumns()) {
            throw new IllegalArgumentException("dimensions are not compatible: (" + A.numRows() + " x " + A.numColumns()
                    + ") cannot be reshaped to (" + rows + " x " + cols + ")");
        }
    }

    static void checkAdd(Dimensions A, Dimensions B, Dimensions C) {
        checkEqualDimension(A, B);
        if (B.numRows() != C.numRows()) {
            throw new IndexOutOfBoundsException(
                    "B.numRows() != C.numRows() (" + B.numRows() + " != " + C.numRows() + ")");
        }
        if (B.numColumns() != C.numColumns()) {
            throw new IndexOutOfBoundsException(
                    "B.numColumns() != C.numColumns() (" + B.numColumns() + " != " + C.numColumns() + ")");
        }
    }

    static void checkMultAdd(Dimensions A, Dimensions B, Dimensions C) {
        if (A.numRows() != C.numRows()) {
            throw new IndexOutOfBoundsException(
                    "A.numRows() != C.numRows() (" + A.numRows() + " != " + C.numRows() + ")");
        }
        if (A.numColumns() != B.numRows()) {
            throw new IndexOutOfBoundsException(
                    "A.numColumns() != B.numRows() (" + A.numColumns() + " != " + B.numRows() + ")");
        }
        if (B.numColumns() != C.numColumns()) {
            throw new IndexOutOfBoundsException(
                    "B.numColumns() != C.numColumns() (" + B.numColumns() + " != " + C.numColumns() + ")");
        }
    }

    static void checkTransABmultAdd(Dimensions A, Dimensions B, Dimensions C) {
        if (A.numRows() != B.numColumns()) {
            throw new IndexOutOfBoundsException(
                    "A.numRows() != B.numColumns() (" + A.numRows() + " != " + B.numColumns() + ")");
        }
        if (A.numColumns() != C.numRows()) {
            throw new IndexOutOfBoundsException(
                    "A.numColumns() != C.numRows() (" + A.numColumns() + " != " + C.numRows() + ")");
        }
        if (B.numRows() != C.numColumns()) {
            throw new IndexOutOfBoundsException(
                    "B.numRows() != C.numColumns() (" + B.numRows() + " != " + C.numColumns() + ")");
        }
    }

    static void checkTransAmultAdd(Dimensions A, Dimensions B, Dimensions C) {
        checkSameRows(A, B);
        if (A.numColumns() != C.numRows()) {
            throw new IndexOutOfBoundsException(
                    "A.numColumns() != C.numRows() (" + A.numColumns() + " != " + C.numRows() + ")");
        }
        if (B.numColumns() != C.numColumns()) {
            throw new IndexOutOfBoundsException(
                    "B.numColumns() != C.numColumns() (" + B.numColumns() + " != " + C.numColumns() + ")");
        }
    }

    static void checkTransBmultAdd(Dimensions A, Dimensions B, Dimensions C) {
        checkSameCols(A, B);
        if (A.numRows() != C.numRows()) {
            throw new IndexOutOfBoundsException(
                    "A.numRows() != C.numRows() (" + A.numRows() + " != " + C.numRows() + ")");
        }
        if (B.numRows() != C.numColumns()) {
            throw new IndexOutOfBoundsException(
                    "B.numRows() != C.numColumns() (" + B.numRows() + " != " + C.numColumns() + ")");
        }
    }

    static void checkSolve(Dimensions A, Dimensions B, Dimensions X) {
        checkSameRows(A, B);
        if (A.numColumns() != X.numRows()) {
            throw new IndexOutOfBoundsException(
                    "A.numColumns() != X.numRows() (" + A.numColumns() + " != " + X.numRows() + ")");
        }
        if (X.numColumns() != B.numColumns()) {
            throw new IndexOutOfBoundsException(
                    "X.numColumns() != B.numColumns() (" + X.numColumns() + " != " + B.numColumns() + ")");
        }
    }

    static void checkSameRows(Dimensions A, Dimensions B) {
        if (A.numRows() != B.numRows()) {
            throw new IndexOutOfBoundsException(
                    "A.numRows() != B.numRows() (" + A.numRows() + " != " + B.numRows() + ")");
        }
    }

    static void checkSameCols(Dimensions A, Dimensions B) {
        if (A.numColumns() != B.numColumns()) {
            throw new IndexOutOfBoundsException(
                    "A.numColumns() != B.numColumns() (" + A.numColumns() + " != " + B.numColumns() + ")");
        }
    }

    static void checkCommensurateColVector(Dimensions matrix, Dimensions colVector) {
        checkSameRows(matrix, colVector);
        if (colVector.numColumns() != 1) {
            throw new IndexOutOfBoundsException(
                    "B is not a column vector (" + colVector.numRows() + " x " + colVector.numColumns() + ")");
        }
    }

    static void checkB_hasAtLeastAsManyRowsAsA(Dimensions A, Dimensions B) {
        if (B.numRows() < A.numRows()) {
            throw new IndexOutOfBoundsException(
                    "B.numRows() < A.numRows() (" + B.numRows() + " < " + A.numRows() + ")");
        }
    }

    static void checkB_hasAtLeastAsManyRowsAsARowColMin(Dimensions A, Dimensions B) {
        if (B.numRows() < Math.min(A.numRows(), A.numColumns())) {
            throw new IndexOutOfBoundsException("B.numRows() < min(A.numRows(), A.numColumns()) (" + B.numRows() + " < "
                    + Math.min(A.numRows(), A.numColumns()) + ")");
        }
    }

    static void checkB_hasAtLeastAsManyColsAsA(Dimensions A, Dimensions B) {
        if (B.numColumns() < A.numColumns()) {
            throw new IndexOutOfBoundsException(
                    "B.numColumns() < A.numColumns() (" + B.numColumns() + " < " + A.numColumns() + ")");
        }
    }

    static void checkB_hasAtLeastAsManyColsAsARowColMin(Dimensions A, Dimensions B) {
        if (B.numColumns() < Math.min(A.numRows(), A.numColumns())) {
            throw new IndexOutOfBoundsException("B.numColumns() < min(A.numRows(), A.numColumns()) (" + B.numColumns()
                    + " < " + Math.min(A.numRows(), A.numColumns()) + ")");
        }
    }

    static void checkMultMany(Dimensions A0, Dimensions A1, Dimensions[] Ai) {
        if (A0.numColumns() != A1.numRows()) {
            throw new IndexOutOfBoundsException(
                    "A0.numColumns() != A1.numRows() (" + A0.numColumns() + " != " + A1.numRows() + ")");
        }
        if (A1.numColumns() != Ai[0].numRows()) {
            throw new IndexOutOfBoundsException(
                    "A1.numColumns() != A2.numRows() (" + A1.numColumns() + " != " + Ai[0].numRows() + ")");
        }
        for (int i = 0; i < Ai.length - 1; ++i) {
            if (Ai[i].numColumns() != Ai[i + 1].numRows()) {
                int colIdx = i + 2;
                int rowIdx = colIdx + 1;
                throw new IndexOutOfBoundsException("A" + colIdx + ".numColumns() != A" + rowIdx + ".numRows() ("
                        + Ai[i].numColumns() + " != " + Ai[i + 1].numRows() + ")");
            }
        }
    }

    static double[] checkJaggedArrayD(double[][] data) {
        int _rows = data.length;
        int _cols = data[0].length;
        if (_rows < 1 || _cols < 1) {
            throw new IllegalArgumentException(
                    "number of rows and columns must be strictly positive : (" + _rows + " x " + _cols + ")");
        }
        return new double[checkArrayLength(_rows, _cols)];
    }

    static float[] checkJaggedArrayF(float[][] data) {
        int _rows = data.length;
        int _cols = data[0].length;
        if (_rows < 1 || _cols < 1) {
            throw new IllegalArgumentException(
                    "number of rows and columns must be strictly positive : (" + _rows + " x " + _cols + ")");
        }
        return new float[checkArrayLength(_rows, _cols)];
    }

    static double[] checkJaggedComplexArrayD(double[][] complexdata) {
        int _rows = complexdata.length;
        int _cols = complexdata[0].length;
        if (_cols % 2 != 0) {
            throw new IllegalArgumentException("complexdata[0].length must be even: " + _cols);
        }
        _cols = _cols / 2;
        if (_rows < 1 || _cols < 1) {
            throw new IllegalArgumentException(
                    "number of rows and columns must be strictly positive : (" + _rows + " x " + _cols + ")");
        }
        return new double[checkComplexArrayLength(_rows, _cols)];
    }

    static float[] checkJaggedComplexArrayF(float[][] complexdata) {
        int _rows = complexdata.length;
        int _cols = complexdata[0].length;
        if (_cols % 2 != 0) {
            throw new IllegalArgumentException("complexdata[0].length must be even: " + _cols);
        }
        _cols = _cols / 2;       
        if (_rows < 1 || _cols < 1) {
            throw new IllegalArgumentException(
                    "number of rows and columns must be strictly positive : (" + _rows + " x " + _cols + ")");
        }
        return new float[checkComplexArrayLength(_rows, _cols)];
    }

    static void checkStdDev(double stdDev) {
        if (stdDev <= 0.0 || Double.isNaN(stdDev) || Double.isInfinite(stdDev)) {
            throw new IllegalArgumentException("Standard deviation must be positive (" + stdDev + ")");
        }
    }

    static void throwInconsistentRowLengths(int cols, int rowIdx, int rowLength) {
        throw new IllegalArgumentException("All rows must have the same length: " + cols + " (row " + rowIdx
                + " has length " + rowLength + ")");
    }

    private Checks() {
        throw new AssertionError();
    }
}
