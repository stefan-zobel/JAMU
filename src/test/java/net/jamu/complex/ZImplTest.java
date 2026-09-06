/*
 * Copyright 2026 Stefan Zobel
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
package net.jamu.complex;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

/**
 * Tests for the mutable complex implementations.
 */
public final class ZImplTest {

    /** the first exponent, the last one and the step of the test ensemble */
    private static final int MIN_EXP_D = -300;
    private static final int MAX_EXP_D = 300;
    private static final int MIN_EXP_F = -30;
    private static final int MAX_EXP_F = 30;
    private static final int ANGLES = 8;

    private static double angle(int k) {
        // 0.3 keeps the components off the axes
        return k * Math.PI / 4.0 + 0.3;
    }

    private static double relative(Zd got, Zd want) {
        return new ZdImpl(got.re() - want.re(), got.im() - want.im()).abs() / want.abs();
    }

    @Test
    public void testInvAgreesWithDivDouble() {
        for (int e = MIN_EXP_D; e <= MAX_EXP_D; ++e) {
            double r = Math.pow(10.0, e);
            for (int k = 0; k < ANGLES; ++k) {
                double re = r * Math.cos(angle(k));
                double im = r * Math.sin(angle(k));
                Zd got = new ZdImpl(re, im).inv();
                Zd want = new ZdImpl(1.0, 0.0).div(new ZdImpl(re, im));
                assertEquals("re at 1e" + e, want.re(), got.re(), 0.0);
                assertEquals("im at 1e" + e, want.im(), got.im(), 0.0);
            }
        }
    }

    @Test
    public void testInvAgreesWithDivFloat() {
        for (int e = MIN_EXP_F; e <= MAX_EXP_F; ++e) {
            float r = (float) Math.pow(10.0, e);
            for (int k = 0; k < ANGLES; ++k) {
                float re = (float) (r * Math.cos(angle(k)));
                float im = (float) (r * Math.sin(angle(k)));
                Zf got = new ZfImpl(re, im).inv();
                Zf want = new ZfImpl(1.0f, 0.0f).div(new ZfImpl(re, im));
                assertEquals("re at 1e" + e, want.re(), got.re(), 0.0f);
                assertEquals("im at 1e" + e, want.im(), got.im(), 0.0f);
            }
        }
    }

    @Test
    public void testInvRoundTrip() {
        for (int e = MIN_EXP_D; e <= MAX_EXP_D; ++e) {
            double r = Math.pow(10.0, e);
            for (int k = 0; k < ANGLES; ++k) {
                Zd z = new ZdImpl(r * Math.cos(angle(k)), r * Math.sin(angle(k)));
                assertTrue("1e" + e, relative(z.copy().inv().inv(), z) <= 1.0e-15);
            }
        }
    }

    @Test
    public void testTimesInverseIsOne() {
        for (int e = MIN_EXP_D; e <= MAX_EXP_D; ++e) {
            double r = Math.pow(10.0, e);
            for (int k = 0; k < ANGLES; ++k) {
                Zd z = new ZdImpl(r * Math.cos(angle(k)), r * Math.sin(angle(k)));
                assertTrue("1e" + e, relative(z.copy().mul(z.copy().inv()), Zd.One()) <= 1.0e-15);
            }
        }
    }

    @Test
    public void testInvWhereTheSquaredModulusWouldOverflow() {
        Zf f = new ZfImpl(1.0e20f, 1.0e20f).inv();
        assertEquals("float re", 5.0e-21f, f.re(), 1.0e-26f);
        assertEquals("float im", -5.0e-21f, f.im(), 1.0e-26f);
        Zd d = new ZdImpl(1.0e200, 1.0e200).inv();
        assertEquals("double re", 5.0e-201, d.re(), 1.0e-215);
        assertEquals("double im", -5.0e-201, d.im(), 1.0e-215);
        Zd u = new ZdImpl(1.0e-200, 1.0e-200).inv();
        assertEquals("double underflow re", 5.0e+199, u.re(), 1.0e+185);
        assertEquals("double underflow im", -5.0e+199, u.im(), 1.0e+185);
    }

    @Test
    public void testInvKeepsTheInfinityConvention() {
        Zd zero = new ZdImpl(0.0, 0.0).inv();
        assertEquals(Double.POSITIVE_INFINITY, zero.re(), 0.0);
        assertEquals(Double.POSITIVE_INFINITY, zero.im(), 0.0);
        Zd inf = new ZdImpl(Double.POSITIVE_INFINITY, 0.0).inv();
        assertEquals(0.0, inf.re(), 0.0);
        assertEquals(0.0, inf.im(), 0.0);
        Zf zeroF = new ZfImpl(0.0f, 0.0f).inv();
        assertEquals(Float.POSITIVE_INFINITY, zeroF.re(), 0.0f);
        assertEquals(Float.POSITIVE_INFINITY, zeroF.im(), 0.0f);
        Zf infF = new ZfImpl(Float.POSITIVE_INFINITY, 0.0f).inv();
        assertEquals(0.0f, infF.re(), 0.0f);
        assertEquals(0.0f, infF.im(), 0.0f);
    }
}
