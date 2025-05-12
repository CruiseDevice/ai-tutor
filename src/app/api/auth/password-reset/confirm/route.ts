// api/auth/password-reset/confirm/route.ts

import { NextResponse } from "next/server";
import { z } from "zod";
import crypto from 'crypto';
import prisma from "@/lib/db";
import bcrypt from 'bcryptjs';

const confirmSchema = z.object({
  token: z.string(),
  password: z.string().min(8)
})

export async function POST(req: Request){
  try {
    const body = await req.json();

    // validate input
    const validation = confirmSchema.safeParse(body);

    if (!validation.success) {
      return NextResponse.json(
        {error: 'Invalid input', details: validation.error.errors},
        {status: 400}
      )
    }

    const { token, password } = validation.data;

    // hash the token to match the hashed token in the database
    const hashedToken = await crypto.createHash('sha256').update(token).digest('hex');
    
    // find the reset token
    const resetToken = await prisma.passwordResetToken.findUnique({
      where: {token: hashedToken}
    })

    if (!resetToken) {
      return NextResponse.json(
        // Delete expired token
        {error: 'Invalid or expired reset token'},
        {status: 400}
      )
    }

    // check if the token is already used
    if (resetToken.used) {
      return NextResponse.json(
        {error: 'Reset token already used'},
        {status: 400}
      )
    }

    // find the user
    const user = await prisma.user.findUnique({
      where: {email: resetToken.email},
    })

    if (!user) {
      return NextResponse.json(
        {error: 'User not found'},
        {status: 404}
      )
    }

    // hash the new password
    const hashedPassword = await bcrypt.hash(password, 10);

    // update user's password and mark token as used
    await prisma.$transaction([
      prisma.user.update({
        where: {id: user.id},
        data: {password: hashedPassword},
      }),
      prisma.passwordResetToken.update({
        where: {id: resetToken.id},
        data: {used: true},
      }),
    ]);

    // Delete all sessions for this user to force re-login
    await prisma.session.deleteMany({
      where: {userId: user.id},
    });

    return NextResponse.json(
      {message: 'Password reset successful'},
      {status: 200}
    );
  } catch (error) {
    console.error('Password reset confirmation error:', error);
    return NextResponse.json(
      {error: 'Internal server error'},
      {status: 500}
    )
  }
}