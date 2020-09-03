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

/**
 * Some basic {@link MatrixD} operations expressed such that the operations'
 * resulting {@code MatrixD} doesn't have to be supplied as an additional
 * parameter.
 */
public interface DMatrixBasicOps {

    /**
     * Creates a submatrix from this matrix that contains all columns from
     * {@code colFrom} (inclusive) up to {@code colTo} (inclusive).
     * 
     * @param colFrom
     *            index of the first column to include in the submatrix
     * @param colTo
     *            index of the last column to include in the submatrix
     * @return a submatrix copy of this matrix that consists of the columns
     *         starting from column index {@code colFrom} up to and including
     *         the column with column index {@code colTo}
     */
    MatrixD selectConsecutiveColumns(int colFrom, int colTo);

    /**
     * {@code A * B} convenience multiplication. None of the operands is
     * mutated.
     * 
     * @param B
     *            second multiplicand
     * @return the result of the multiplication
     */
    MatrixD times(MatrixD B);

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
    MatrixD timesTimes(MatrixD B, MatrixD C);

    /**
     * {@code A + B} convenience addition. None of the operands is mutated.
     * 
     * @param B
     *            the addend
     * @return the result of the addition
     */
    MatrixD plus(MatrixD B);

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
    MatrixD timesPlus(MatrixD B, MatrixD C);

    /**
     * {@code A - B} convenience subtraction. None of the operands is mutated.
     * 
     * @param B
     *            the subtrahend
     * @return the result of the subtraction
     */
    MatrixD minus(MatrixD B);

    /**
     * Unary minus {@code -A} convenience method. None of the operands is
     * mutated.
     * 
     * @return {@code -A}
     */
    MatrixD uminus();

    /**
     * Returns <code>A<sup>T</sup></code>. None of the operands is mutated.
     * 
     * @return the transposed matrix
     */
    MatrixD transpose();

    /**
     * Returns <code>A<sup>-1</sup></code> for quadratic matrices. None of the
     * operands is mutated.
     * 
     * @return the inverse of this matrix if it is quadratic
     * @throws IllegalArgumentException if this matrix is not quadratic
     */
    MatrixD inverse();
}
