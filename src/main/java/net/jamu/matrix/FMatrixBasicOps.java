/*
 * Copyright 2020 Stefan Zobel
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
 * Some basic {@link MatrixF} operations expressed such that the operations'
 * resulting {@code MatrixF} doesn't have to be supplied as an additional
 * parameter.
 */
public interface FMatrixBasicOps {

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
     * {@code A \ B} matrix left division. {@code X = A\B} is the solution to
     * the equation {@code A * X = B}.
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
}
