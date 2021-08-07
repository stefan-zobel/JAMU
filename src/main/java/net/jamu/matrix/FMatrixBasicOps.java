/*
 * Copyright 2020, 2021 Stefan Zobel
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

import net.frobenius.ComputationTruncatedException;

/**
 * Some basic operations for {@link MatrixF} objects expressed such that the
 * operations' resulting {@code MatrixF} doesn't have to be supplied as an
 * additional parameter. None of these operations mutate the receiving matrix
 * instance. In some sense these are all convenience methods, which could be
 * expressed by more fundamental methods from {@link MatrixF} leading to more
 * cumbersome code
 */
public interface FMatrixBasicOps {

    /**
     * Creates a column vector copy from this matrix that contains the column at
     * index {@code col}.
     * 
     * @param col
     *            zero based index of the column to return
     * @return a column vector that contains the column indexed by {@code col}
     *         of this matrix
     * @since 1.2
     */
    MatrixF selectColumn(int col);

    /**
     * Creates a submatrix copy from this matrix that contains all columns from
     * {@code colFrom} (inclusive) up to {@code colTo} (inclusive).
     * 
     * @param colFrom
     *            index (zero based) of the first column to include in the
     *            submatrix
     * @param colTo
     *            index (zero based) of the last column to include in the
     *            submatrix
     * @return a submatrix copy of this matrix that consists of the columns
     *         starting from column index {@code colFrom} up to and including
     *         the column with column index {@code colTo}
     */
    MatrixF selectConsecutiveColumns(int colFrom, int colTo);

    /**
     * Creates a submatrix copy of dimension
     * {@code (rowTo - rowFrom + 1) x (colTo - colFrom + 1)} from this matrix
     * that contains the submatrix which has {@code (rowFrom, colFrom)} as its
     * upper left corner and {@code (rowTo, colTo)} as its lower right corner.
     * Indices are zero-based.
     * 
     * @param rowFrom
     *            index (zero based) of the first row to include in the
     *            submatrix (left upper corner)
     * @param colFrom
     *            index (zero based) of the first column to include in the
     *            submatrix (left upper corner)
     * @param rowTo
     *            index (zero based) of the last row to include in the submatrix
     *            (right lower corner)
     * @param colTo
     *            index (zero based) of the last column to include in the
     *            submatrix (right lower corner)
     * @return a submatrix copy of this matrix that contains the submatrix with
     *         upper left corner {@code (rowFrom, colFrom)} and lower right
     *         corner {@code (rowTo, colTo)}
     * @since 1.2
     */
    MatrixF selectSubmatrix(int rowFrom, int colFrom, int rowTo, int colTo);

    /**
     * Appends a column vector which must have dimension
     * {@code this.numRows() x 1} to the right of this matrix returning a new
     * matrix of dimension {@code this.numRows() x (this.numColumns() + 1)}
     * where the values from the passed column vector occupy the newly created
     * column at column index {@code this.numColumns()} and all other values
     * from this matrix have been copied over to the new matrix. This matrix
     * doesn't get mutated by the append operation.
     * 
     * @param colVector
     *            the column vector to append to the right of this matrix. Must
     *            have dimension {@code this.numRows() x 1}
     * @return a newly created copy of this matrix with one more column appended
     *         to the right which contains the contents of the provided
     *         {@code colVector} argument
     */
    MatrixF appendColumn(MatrixF colVector);

    /**
     * Appends a matrix which must have dimension {@code this.numRows() x k} to
     * the right of this matrix returning a new matrix of dimension
     * {@code this.numRows() x (this.numColumns() + k)} where the values from
     * the passed matrix occupy the newly created columns starting at column
     * index {@code this.numColumns()} and all other values from this matrix
     * have been copied over to the new matrix. This matrix doesn't get mutated
     * by the append operation.
     * 
     * @param matrix
     *            the matrix to append to the right of this matrix. Must have
     *            dimension {@code this.numRows() x k}
     * @return a newly created copy of this matrix with the columns from
     *         {@code matrix} appended to the right
     * @since 1.2
     */
    MatrixF appendMatrix(MatrixF matrix);

    /**
     * {@code A \ B} matrix left division. {@code X = A\B} is the solution to
     * the equation {@code A * X = B}.
     * <p>
     * Note: Matrix left division and matrix right division are related by the
     * equation <code>(B<sup>T</sup> \ A<sup>T</sup>)<sup>T</sup> = A / B</code>
     * 
     * @param B
     *            a matrix that has the same number of rows as this matrix
     *            ({@code A})
     * @return the result of the left division
     * @throws ComputationTruncatedException
     *             for exactly singular factors in the LU decomposition of a
     *             quadratic matrix or for a non-quadratic matrix that doesn't
     *             have full rank
     */
    MatrixF mldivide(MatrixF B);

    /**
     * {@code A / B} matrix right division. {@code X = A/B} is the solution to
     * the equation {@code X * B = A}.
     * <p>
     * Note: Matrix right division and matrix left division are related by the
     * equation <code>A / B = (B<sup>T</sup> \ A<sup>T</sup>)<sup>T</sup></code>
     * 
     * @param B
     *            a matrix that has the same number of columns as this matrix
     *            ({@code A})
     * @return the result of the right division
     * @throws ComputationTruncatedException
     *             for exactly singular factors in the LU decomposition of a
     *             quadratic matrix or for a non-quadratic matrix that doesn't
     *             have full rank
     */
    MatrixF mrdivide(MatrixF B);

    /**
     * {@code A * B} convenience multiplication. None of the operands is
     * mutated.
     * 
     * @param B
     *            second multiplicand
     * @return the result of the multiplication
     */
    MatrixF times(MatrixF B);

    /**
     * {@code A * B * C} convenience multiplication. None of the operands is
     * mutated.
     * 
     * @param B
     *            second multiplicand
     * @param C
     *            third multiplicand
     * @return the result of the multiplication
     */
    MatrixF timesTimes(MatrixF B, MatrixF C);

    /**
     * <code>A * A<sup>T</sup></code> multiplication. This is much more
     * efficient than the equivalent {@code A.times(A.transpose())}. None of the
     * operands is mutated.
     * 
     * @return the result of the multiplication
     * @since 1.2.1
     */
    MatrixF timesTransposed();

    /**
     * Multiply this matrix {@code A} with a complex matrix {@code B} returning
     * the result of the multiplication {@code A * B} as a complex matrix. None
     * of the operands is mutated.
     * 
     * @param B
     *            second multiplicand (a complex matrix)
     * @return the result of the multiplication with the complex matrix
     */
    ComplexMatrixF times(ComplexMatrixF B);

    /**
     * {@code A + B} convenience addition. None of the operands is mutated.
     * 
     * @param B
     *            the addend
     * @return the result of the addition
     */
    MatrixF plus(MatrixF B);

    /**
     * {@code A * B + C} convenience multiplication plus addition. None of the
     * operands is mutated.
     * 
     * @param B
     *            second multiplicand
     * @param C
     *            the addend
     * @return the result of the two operations
     */
    MatrixF timesPlus(MatrixF B, MatrixF C);

    /**
     * {@code A - B} convenience subtraction. None of the operands is mutated.
     * 
     * @param B
     *            the subtrahend
     * @return the result of the subtraction
     */
    MatrixF minus(MatrixF B);

    /**
     * Unary minus {@code -A} convenience method. None of the operands is
     * mutated.
     * 
     * @return {@code -A}
     */
    MatrixF uminus();

    /**
     * Returns {@code |A|}, i.e. a matrix where all elements
     * <code>a<sub>ij</sub></code> have been replaced by their absolute value
     * <code>|a<sub>ij</sub>|</code>. None of the operands is mutated.
     * 
     * @return {@code |A|}, the matrix of absolute values of {@code A}
     */
    MatrixF abs();

    /**
     * Returns <code>A<sup>T</sup></code>. None of the operands is mutated.
     * 
     * @return the transposed matrix
     */
    MatrixF transpose();

    /**
     * Returns <code>A<sup>-1</sup></code> for quadratic matrices. None of the
     * operands is mutated.
     * 
     * @return the inverse of this matrix if it is quadratic
     * @throws IllegalArgumentException
     *             if this matrix is not quadratic
     */
    MatrixF inverse();

    /**
     * Reshapes this matrix into a new matrix of dimension {@code rows x cols}
     * where the elements in this matrix are read in Fortran-style column-major
     * order. For example, the {@code 3 x 2} matrix {@code A}
     * 
     * <pre>
     * <code>
     *     1 4
     * A = 2 5
     *     3 6
     * </code>
     * </pre>
     * 
     * reshaped to a {@code 2 x 3} matrix {@code B} (i.e.,
     * {@code B = A.reshape(2, 3);}) would become
     * 
     * <pre>
     * <code>
     * B = 1 3 5
     *     2 4 6
     * </code>
     * </pre>
     * 
     * The new shape must be compatible with the original shape in the sense
     * that {@code rows x cols == this.numRows() x this.numColumns()} is
     * required, otherwise an {@code IllegalArgumentException} is thrown. None
     * of the operands is mutated.
     * 
     * @param rows
     *            the desired number of rows of the reshaped matrix
     * @param cols
     *            the desired number of columns of the reshaped matrix
     * @return the reshaped matrix
     * @throws IllegalArgumentException
     *             if {@code rows x cols != this.numRows() x this.numColumns()}
     */
    MatrixF reshape(int rows, int cols);

    /**
     * Convert this matrix to a complex matrix, keeping the real parts with
     * all imaginary parts set to 0.0f.
     * 
     * @return this matrix converted to a complex matrix (all imaginary parts
     *         get initialized with 0.0f)
     */
    ComplexMatrixF toComplexMatrix();
}
