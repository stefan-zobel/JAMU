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

/**
 * Basic matrix properties mostly related to the dimension of a matrix.
 */
public interface Dimensions {

    /**
     * {@code true} if and only if this matrix is a 1-by-1 matrix.
     * 
     * @return {@code true} if this matrix can be converted to a scalar,
     *         {@code false} otherwise
     */
    boolean isScalar();

    /**
     * {@code true} if and only if this matrix is a n-by-1 matrix.
     * 
     * @return {@code true} if this matrix is a column vector, {@code false}
     *         otherwise
     */
    boolean isColumnVector();

    /**
     * {@code true} if and only if this matrix is a 1-by-n matrix.
     * 
     * @return {@code true} if this matrix is a row vector, {@code false}
     *         otherwise
     */
    boolean isRowVector();

    /**
     * {@code true} if and only if this matrix is a square n-by-n matrix.
     * 
     * @return {@code true} if this matrix is a quadratic matrix, {@code false}
     *         otherwise
     */
    boolean isSquareMatrix();

    /**
     * Returns the number of columns (&gt;= 1) of this matrix.
     * 
     * @return the number of matrix columns
     */
    int numColumns();

    /**
     * Returns the number of rows (&gt;= 1) of this matrix.
     * 
     * @return the number of matrix rows
     */
    int numRows();

    /**
     * Returns the first valid row index of this matrix (which is always
     * {@code 0}).
     * 
     * @return the index of the first row of this matrix
     */
    int startRow();

    /**
     * Returns the last valid row index of this matrix (which is always
     * {@code numRows() - 1}).
     * 
     * @return the index of the last row of this matrix
     */
    int endRow();

    /**
     * Returns the first valid column index of this matrix (which is always
     * {@code 0}).
     * 
     * @return the index of the first column of this matrix
     */
    int startCol();

    /**
     * Returns the last valid column index of this matrix (which is always
     * {@code numColumns() - 1}).
     * 
     * @return the index of the last column of this matrix
     */
    int endCol();

    /**
     * {@code true} if and only if this matrix is a complex matrix.
     * 
     * @return {@code true} if this matrix is a complex matrix, {@code false}
     *         otherwise
     * @since 1.3
     */
    boolean isComplex();

    /**
     * {@code true} if and only if this matrix holds double precision values.
     * 
     * @return {@code true} if this matrix holds double precision values,
     *         {@code false} if it holds single precision values
     * @since 1.3
     */
    boolean isDoublePrecision();

    /**
     * Check that {@code row} and {@code col} are valid (zero-based) indexes for
     * this matrix. Throws {@link IllegalArgumentException} otherwise.
     * 
     * @param row
     *            zero-based row index
     * @param col
     *            zero-based column index
     * @throws IllegalArgumentException
     *             if {@code (row, col)} is an invalid matrix index
     */
    void checkIndex(int row, int col);

    /**
     * {@code (rFrom, cFrom)} upper left corner, {@code (rTo, cTo)} lower right
     * corner. All indexes must be valid and the submatrix must contain at least
     * one element.
     * 
     * @param rFrom
     *            {@code <= rTo}
     * @param cFrom
     *            {@code <= cTo}
     * @param rTo
     *            {@code >= rFrom}
     * @param cTo
     *            {@code >= rTo}
     */
    void checkSubmatrixIndexes(int rFrom, int cFrom, int rTo, int cTo);

    /**
     * Returns a string representation of this {@code Dimensions} instance.
     * 
     * @return a string representation of the matrix dimensions
     */
    String asString();
}
