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
import net.jamu.complex.Zf;

/**
 * A {@code ComplexMatrixF} is a dense matrix of single precision complex
 * numbers expressed as an array of primitive floats with column-major storage
 * layout. The addressing is zero based. All operations throw a
 * {@code NullPointerException} if any of the method arguments is {@code null}.
 */
public interface ComplexMatrixF extends Dimensions, FComplexMatrixBasicOps {

    /**
     * Get the single element as a scalar if this matrix is 1-by-1.
     * 
     * @return the single element as a scalar if this matrix is 1-by-1
     * @throws IllegalStateException
     *             if this matrix is not 1-by-1
     */
    Zf toScalar();

    /**
     * {@code A = alpha * A}
     * 
     * @param alphar
     *            real part of the scaling factor
     * @param alphai
     *            imaginary part of the scaling factor
     * @return {@code A}
     */
    ComplexMatrixF scaleInplace(float alphar, float alphai);

    /**
     * {@code B = alpha * A}
     * 
     * @param alphar
     *            real part of the scaling factor
     * @param alphai
     *            imaginary part of the scaling factor
     * @param B
     *            output matrix
     * @return {@code B}
     */
    ComplexMatrixF scale(float alphar, float alphai, ComplexMatrixF B);

    /**
     * Stores <code>AH = A<sup>*</sup></code> (i.e., the conjugate transpose of
     * {@code A}) in {@code AH}.
     * 
     * @param AH
     *            output matrix (mutated)
     * @return {@code AH}
     */
    ComplexMatrixF conjTrans(ComplexMatrixF AH);

    /**
     * Stores <code>AT = A<sup>T</sup></code> (i.e., the transpose of {@code A})
     * in {@code AT}.
     * 
     * @param AT
     *            output matrix (mutated)
     * @return {@code AT}
     */
    ComplexMatrixF trans(ComplexMatrixF AT);

    /**
     * {@code A = A + B}
     * 
     * @param B
     *            the matrix to be added to this matrix
     * @return {@code A}
     */
    ComplexMatrixF addInplace(ComplexMatrixF B);

    /**
     * {@code A = A + alpha * B}
     * 
     * @param alphar
     *            real part of the scaling factor for {@code B}
     * @param alphai
     *            imaginary part of the scaling factor for {@code B}
     * @param B
     *            matrix to be added to this matrix (after scaling)
     * @return {@code A}
     */
    ComplexMatrixF addInplace(float alphar, float alphai, ComplexMatrixF B);

    /**
     * {@code C = A + B}
     * 
     * @param B
     *            matrix to be added to this matrix
     * @param C
     *            output matrix for the result
     * @return {@code C}
     */
    ComplexMatrixF add(ComplexMatrixF B, ComplexMatrixF C);

    /**
     * {@code C = A + alpha * B}
     * 
     * @param alphar
     *            real part of the scaling factor for {@code B}
     * @param alphai
     *            imaginary part of the scaling factor for {@code B}
     * @param B
     *            matrix to be added to this matrix (after scaling)
     * @param C
     *            output matrix for the result
     * @return {@code C}
     */
    ComplexMatrixF add(float alphar, float alphai, ComplexMatrixF B, ComplexMatrixF C);

    /**
     * {@code C = A * B}
     * 
     * @param B
     *            matrix to be multiplied from the right
     * @param C
     *            output matrix for the result of the multiplication
     * @return {@code C}
     */
    ComplexMatrixF mult(ComplexMatrixF B, ComplexMatrixF C);

    /**
     * {@code C = alpha * A * B}
     * 
     * @param alphar
     *            real part of the scaling factor for the multiplication
     * @param alphai
     *            imaginary part of the scaling factor for the multiplication
     * @param B
     *            matrix to be multiplied from the right
     * @param C
     *            output matrix for the result of the multiplication
     * @return {@code C}
     */
    ComplexMatrixF mult(float alphar, float alphai, ComplexMatrixF B, ComplexMatrixF C);

    /**
     * {@code C = A * B + C}. On exit, the matrix {@code C} is overwritten by
     * the result of the operation.
     * 
     * @param B
     *            matrix to be multiplied from the right
     * @param C
     *            the matrix to add on input, contains the result of the
     *            operation on output
     * @return {@code C}
     */
    ComplexMatrixF multAdd(ComplexMatrixF B, ComplexMatrixF C);

    /**
     * {@code C = alpha * A * B + C}. On exit, the matrix {@code C} is
     * overwritten by the result of the operation.
     * 
     * @param alphar
     *            real part of the scaling factor for the multiplication
     * @param alphai
     *            imaginary part of the scaling factor for the multiplication
     * @param B
     *            matrix to be multiplied from the right
     * @param C
     *            the matrix to add on input, contains the result of the
     *            operation on output
     * @return {@code C}
     */
    ComplexMatrixF multAdd(float alphar, float alphai, ComplexMatrixF B, ComplexMatrixF C);

    /**
     * <code>C = A<sup>*</sup> * B<sup>*</sup></code>
     * 
     * @param B
     *            matrix whose conjugate transpose is to be multiplied from the
     *            right
     * @param C
     *            output matrix for the result of the multiplication
     * @return {@code C}
     */
    ComplexMatrixF conjTransABmult(ComplexMatrixF B, ComplexMatrixF C);

    /**
     * <code>C = alpha * A<sup>*</sup> * B<sup>*</sup></code>
     * 
     * @param alphar
     *            real part of the scaling factor for the multiplication
     * @param alphai
     *            imaginary part of the scaling factor for the multiplication
     * @param B
     *            matrix whose conjugate transpose is to be multiplied from the
     *            right
     * @param C
     *            output matrix for the result of the multiplication
     * @return {@code C}
     */
    ComplexMatrixF conjTransABmult(float alphar, float alphai, ComplexMatrixF B, ComplexMatrixF C);

    /**
     * <code>C = A<sup>*</sup> * B</code>
     * 
     * @param B
     *            matrix to be multiplied from the right
     * @param C
     *            output matrix for the result of the multiplication
     * @return {@code C}
     */
    ComplexMatrixF conjTransAmult(ComplexMatrixF B, ComplexMatrixF C);

    /**
     * <code>C = alpha * A<sup>*</sup> * B</code>
     * 
     * @param alphar
     *            real part of the scaling factor for the multiplication
     * @param alphai
     *            imaginary part of the scaling factor for the multiplication
     * @param B
     *            matrix to be multiplied from the right
     * @param C
     *            output matrix for the result of the multiplication
     * @return {@code C}
     */
    ComplexMatrixF conjTransAmult(float alphar, float alphai, ComplexMatrixF B, ComplexMatrixF C);

    /**
     * <code>C = A * B<sup>*</sup></code>
     * 
     * @param B
     *            matrix whose conjugate transpose is to be multiplied from the
     *            right
     * @param C
     *            output matrix for the result of the multiplication
     * @return {@code C}
     */
    ComplexMatrixF conjTransBmult(ComplexMatrixF B, ComplexMatrixF C);

    /**
     * <code>C = alpha * A * B<sup>*</sup></code>
     * 
     * @param alphar
     *            real part of the scaling factor for the multiplication
     * @param alphai
     *            imaginary part of the scaling factor for the multiplication
     * @param B
     *            matrix whose conjugate transpose is to be multiplied from the
     *            right
     * @param C
     *            output matrix for the result of the multiplication
     * @return {@code C}
     */
    ComplexMatrixF conjTransBmult(float alphar, float alphai, ComplexMatrixF B, ComplexMatrixF C);

    /**
     * <code>C = A<sup>*</sup> * B<sup>*</sup> + C</code>. On exit, the matrix
     * {@code C} is overwritten by the result of the operation.
     * 
     * @param B
     *            matrix whose conjugate transpose is to be multiplied from the
     *            right
     * @param C
     *            the matrix to add on input, contains the result of the
     *            operation on output
     * @return {@code C}
     */
    ComplexMatrixF conjTransABmultAdd(ComplexMatrixF B, ComplexMatrixF C);

    /**
     * <code>C = alpha * A<sup>*</sup> * B<sup>*</sup> + C</code>. On exit, the
     * matrix {@code C} is overwritten by the result of the operation.
     * 
     * @param alphar
     *            real part of the scaling factor for the multiplication
     * @param alphai
     *            imaginary part of the scaling factor for the multiplication
     * @param B
     *            matrix whose conjugate transpose is to be multiplied from the
     *            right
     * @param C
     *            the matrix to add on input, contains the result of the
     *            operation on output
     * @return {@code C}
     */
    ComplexMatrixF conjTransABmultAdd(float alphar, float alphai, ComplexMatrixF B, ComplexMatrixF C);

    /**
     * <code>C = A<sup>*</sup> * B + C</code>. On exit, the matrix {@code C} is
     * overwritten by the result of the operation.
     * 
     * @param B
     *            matrix to be multiplied from the right
     * @param C
     *            the matrix to add on input, contains the result of the
     *            operation on output
     * @return {@code C}
     */
    ComplexMatrixF conjTransAmultAdd(ComplexMatrixF B, ComplexMatrixF C);

    /**
     * <code>C = alpha * A<sup>*</sup> * B + C</code>. On exit, the matrix
     * {@code C} is overwritten by the result of the operation.
     * 
     * @param alphar
     *            real part of the scaling factor for the multiplication
     * @param alphai
     *            imaginary part of the scaling factor for the multiplication
     * @param B
     *            matrix to be multiplied from the right
     * @param C
     *            the matrix to add on input, contains the result of the
     *            operation on output
     * @return {@code C}
     */
    ComplexMatrixF conjTransAmultAdd(float alphar, float alphai, ComplexMatrixF B, ComplexMatrixF C);

    /**
     * <code>C = A * B<sup>*</sup> + C</code>. On exit, the matrix {@code C} is
     * overwritten by the result of the operation.
     * 
     * @param B
     *            matrix whose conjugate transpose is to be multiplied from the
     *            right
     * @param C
     *            the matrix to add on input, contains the result of the
     *            operation on output
     * @return {@code C}
     */
    ComplexMatrixF conjTransBmultAdd(ComplexMatrixF B, ComplexMatrixF C);

    /**
     * <code>C = alpha * A * B<sup>*</sup> + C</code>. On exit, the matrix
     * {@code C} is overwritten by the result of the operation.
     * 
     * @param alphar
     *            real part of the scaling factor for the multiplication
     * @param alphai
     *            imaginary part of the scaling factor for the multiplication
     * @param B
     *            matrix whose conjugate transpose is to be multiplied from the
     *            right
     * @param C
     *            the matrix to add on input, contains the result of the
     *            operation on output
     * @return {@code C}
     */
    ComplexMatrixF conjTransBmultAdd(float alphar, float alphai, ComplexMatrixF B, ComplexMatrixF C);

    /**
     * Get a newly created copy of this matrix.
     * 
     * @return fresh copy of this matrix
     */
    ComplexMatrixF copy();

    /**
     * Set all elements of this matrix to {@code 0.0f + i * 0.0f} mutating this
     * matrix.
     * 
     * @return this matrix (mutated)
     */
    ComplexMatrixF zeroInplace();

    /**
     * Copy the {@code other} matrix into this matrix (mutating this matrix)
     * where the dimensions of {@code other} and {@code this} must be the same.
     * 
     * @param other
     *            matrix whose elements should be copied into this matrix
     * @return this matrix (mutated)
     */
    ComplexMatrixF setInplace(ComplexMatrixF other);

    /**
     * {@code A = alpha * B}
     * 
     * @param alphar
     *            the real part of the scale factor for {@code B}
     * @param alphai
     *            the imaginary part of the scale factor for {@code B}
     * @param other
     *            matrix to be copied into this matrix after the scalar
     *            multiplication
     * @return {@code A}
     */
    ComplexMatrixF setInplace(float alphar, float alphai, ComplexMatrixF other);

    /**
     * Copy the matrix element at {@code (row, col)} into {@code out}.
     * 
     * @param row
     *            row index, zero-based
     * @param col
     *            column index, zero-based
     * 
     * @param out
     *            receiver argument (mutated)
     */
    void get(int row, int col, Zf out);

    /**
     * Get the matrix element at {@code (row, col)}.
     * 
     * @param row
     *            row index, zero-based
     * @param col
     *            column index, zero-based
     * @return the matrix element at {@code (row, col)}
     */
    Zf get(int row, int col);

    /**
     * Set the matrix element at {@code (row, col)} to {@code val} mutating this
     * matrix.
     * 
     * @param row
     *            row index, zero-based
     * @param col
     *            column index, zero-based
     * @param valr
     *            the real part of the new value
     * @param vali
     *            the imaginary part of the new value
     * @return this matrix (mutated)
     */
    ComplexMatrixF set(int row, int col, float valr, float vali);

    /**
     * Add {@code val} to the matrix element at {@code (row, col)} mutating this
     * matrix.
     * 
     * @param row
     *            row index, zero-based
     * @param col
     *            column index, zero-based
     * @param valr
     *            the real part of the value to add to the element at
     *            {@code (row, col)}
     * @param vali
     *            the imaginary part of the value to add to the element at
     *            {@code (row, col)}
     * @return this matrix (mutated)
     */
    ComplexMatrixF add(int row, int col, float valr, float vali);

    /**
     * Copy a submatrix of this matrix into {@code B}.
     * 
     * @param r0
     *            initial row index (left upper corner) in this matrix
     * @param c0
     *            initial col index (left upper corner) in this matrix
     * @param r1
     *            last row index (right lower corner) in this matrix
     * @param c1
     *            last col index (right lower corner) in this matrix
     * @param B
     *            matrix of dimension at least
     *            {@code (r1 - r0 + 1) x (c1 - c0 + 1)}
     * @param rb
     *            initial row index (left upper corner) in the matrix {@code B}
     * @param cb
     *            initial col index (left upper corner) in the matrix {@code B}
     * @return the submatrix {@code B}
     */
    ComplexMatrixF submatrix(int r0, int c0, int r1, int c1, ComplexMatrixF B, int rb, int cb);

    /**
     * Set a submatrix from the values of matrix {@code B} extending from
     * {@code (rb0, cb0)} to {@code (rb1, cb1)} (the upper left and lower right
     * corner in {@code B} respectively) at position {@code (r0, c0)} in this
     * matrix.
     * 
     * @param r0
     *            initial row index (left upper corner) in this matrix
     * @param c0
     *            initial col index (left upper corner) in this matrix
     * @param rb0
     *            initial row index (left upper corner) in the matrix {@code B}
     * @param cb0
     *            initial col index (left upper corner) in the matrix {@code B}
     * @param rb1
     *            last row index (right lower corner) in the matrix {@code B}
     * @param cb1
     *            last col index (right lower corner) in the matrix {@code B}
     * @param B
     *            the matrix that holds the values to set in this matrix
     * @return this matrix {@code A}
     */
    ComplexMatrixF setSubmatrixInplace(int r0, int c0, ComplexMatrixF B, int rb0, int cb0, int rb1, int cb1);

    /**
     * Computes the solution ({@code X}) to a system of linear equations
     * {@code A * X = B}, where {@code A} is either a {@code n x n} matrix and
     * {@code X} and {@code B} are {@code n x r} matrices, or where {@code A} is
     * a {@code n x m} and matrix {@code X} is a {@code m x r} matrix and
     * {@code B} is a {@code n x r} matrix.
     * 
     * @param B
     *            matrix with the same number of rows as this matrix {@code A},
     *            and the same number of columns as {@code X}
     * @param X
     *            matrix with number of rows equal to the number of columns of
     *            this matrix {@code A}, and the same number of columns as
     *            {@code B}
     * @return {@code X}, the solution of dimension either {@code n x r} (in the
     *         {@code n x n} case) or {@code m x r} (in the {@code m x n} case).
     * @throws ComputationTruncatedException
     *             for exactly singular factors in the LU decomposition of a
     *             quadratic matrix or for a non-quadratic matrix that doesn't
     *             have full rank
     */
    ComplexMatrixF solve(ComplexMatrixF B, ComplexMatrixF X);

    /**
     * Matrix inverse for quadratic matrices.
     * 
     * @param inverse
     *            matrix where the inverse is stored. Must have the same
     *            dimension as this matrix
     * @return the inverse matrix (i.e. the argument {@code inverse})
     * @throws IllegalArgumentException
     *             if this matrix is not quadratic or if {@code inverse} has the
     *             wrong dimension
     * @throws ComputationTruncatedException
     *             for exactly singular factors in the LU decomposition of this
     *             matrix
     */
    ComplexMatrixF inv(ComplexMatrixF inverse);

    /**
     * Computes the eigenvalue decomposition of this matrix if it is quadratic.
     * 
     * @param full
     *            controls whether the (right) eigenvectors should be computed
     *            in addition (if {@code true}) or the eigenvalues only (if
     *            {@code false})
     * @return the {@link EvdComplexF} of this matrix, either full or the
     *         eigenvalues only (if {@code full} is set to {@code false})
     * @throws IllegalArgumentException
     *             if this matrix is not quadratic
     * @throws ComputationTruncatedException
     *             if the QR decomposition failed to compute all eigenvalues
     */
    EvdComplexF evd(boolean full);

    // TODO ...

    /**
     * Matrix trace of a square matrix.
     * 
     * @return sum of the diagonal elements
     * @throws IllegalArgumentException
     *             if this matrix is not quadratic
     */
    Zf trace();

    /**
     * Get the reference to the internal backing array without copying.
     * 
     * @return the reference to the internal backing array
     */
    float[] getArrayUnsafe();

    /**
     * Copy the matrix element {@code (row, col)} without bounds checking into
     * {@code out}.
     * 
     * @param row
     *            row index, zero-based
     * @param col
     *            column index, zero-based
     * @param out
     *            receiver argument (mutated)
     */
    void getUnsafe(int row, int col, Zf out);

    /**
     * Get the matrix element {@code (row, col)} without bounds checking.
     * 
     * @param row
     *            row index, zero-based
     * @param col
     *            column index, zero-based
     * @return the matrix element at {@code (row, col)}
     */
    Zf getUnsafe(int row, int col);

    /**
     * Set the matrix element at {@code (row, col)} to {@code val} without
     * bounds checking.
     * 
     * @param row
     *            row index, zero-based
     * @param col
     *            column index, zero-based
     * @param valr
     *            real part of the new value
     * @param vali
     *            imaginary part of the new value
     */
    void setUnsafe(int row, int col, float valr, float vali);
}