import { NextResponse } from "next/server";
import prisma from "@/lib/db";
import { z } from "zod";
import crypto from 'crypto';

const requestSchema = z.object({
  email: z.string().email()
})


export async function POST(req: Request) {
  try {
    const body = await req.json();

    // validate input
    const validation = requestSchema.safeParse(body);
    if (!validation.success) {
      return NextResponse.json(
        {error: 'Invalid email', details: validation.error.errors},
        {status: 400}
      );
    }

    const { email } = validation.data;

    // Check if user exists
    const user = await prisma.user.findUnique({
      where: { email }
    });

    // Always return success even if user doesn't exist (security best practice)
    if (!user) {
      return NextResponse.json(
        { message: 'If an account with that email exists, you will receive a password reset link.' },
        { status: 200 }
      );
    }

    // generate reset token
    const resetToken = crypto.randomBytes(32).toString('hex');
    const hashedToken = await crypto.createHash('sha256').update(resetToken).digest('hex');

    // set expiration time (1 hour from now)
    const expiresAt = new Date(Date.now() + 3600000);

    // Delete any existing reset tokens for this email
    await prisma.passwordResetToken.deleteMany({
      where: { email }
    });

    // Create new reset token
    await prisma.passwordResetToken.create({
      data: {
        email,
        token: hashedToken,
        expiresAt,
      },
    });

    console.log('Reset token created for email:', email);
    console.log('Reset URL:', `${req.headers.get('origin')}/reset-password?token=${resetToken}`);

    return NextResponse.json(
      {message: 'If an account with that email exists, you will receive a password reset link.'},
      {status: 200}
    );

  } catch (error) {
    console.error('Password reset request error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}


