/*
 * Copyright 2021 Stefan Zobel
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
 * Packed upper triangular matrix. Packed in a 1-dimensional array in
 * column-major order. Indexing is zero-based.
 */
abstract class UpperPacked {

    final int n;

    UpperPacked(int dim) {
        if (dim <= 0) {
            throw new IllegalArgumentException("dim: " + dim);
        }
        n = dim;
    }

    final int idx(int row, int col) {
        if (row <= col) {
            return (col * (col + 1)) / 2 + row;
        }
        // below the main diagonal
        throw new IllegalArgumentException(row + "," + col);
    }

    final int length() {
        return (n * (n + 1)) / 2;
    }

    final int dim() {
        return n;
    }

    // for debugging only
    public String toString() {
        StringBuilder buf = new StringBuilder();
        for (int row = 0; row < n; ++row) {
            for (int col = 0; col < n; ++col) {
                if (col < row) {
                    buf.append("- ");
                } else {
                    buf.append(getVal(row, col)).append(" ");
                }
            }
            buf.append("\n");
        }
        return buf.toString();
    }

    // for toString() only
    abstract Object getVal(int row, int col);
}
