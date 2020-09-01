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
 * Some basic {@link ComplexMatrixD} operations expressed such that the
 * operations' resulting {@code ComplexMatrixD} doesn't have to be supplied as
 * an additional parameter.
 */
public interface DComplexMatrixBasicOps {

    /**
     * {@code A * B} convenience multiplication. None of the operands is
     * mutated.
     * 
     * @param B
     *            second multiplicand
     * @return the result of the multiplication
     */
    ComplexMatrixD times(ComplexMatrixD B);

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
    ComplexMatrixD timesTimes(ComplexMatrixD B, ComplexMatrixD C);

    /**
     * {@code A + B} convenience addition. None of the operands is mutated.
     * 
     * @param B
     *            the addend
     * @return the result of the addition
     */
    ComplexMatrixD plus(ComplexMatrixD B);

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
    ComplexMatrixD timesPlus(ComplexMatrixD B, ComplexMatrixD C);

    /**
     * {@code A - B} convenience subtraction. None of the operands is mutated.
     * 
     * @param B
     *            the subtrahend
     * @return the result of the subtraction
     */
    ComplexMatrixD minus(ComplexMatrixD B);

    /**
     * Unary minus {@code -A} convenience method. None of the operands is
     * mutated.
     * 
     * @return {@code -A}
     */
    ComplexMatrixD uminus();
}
