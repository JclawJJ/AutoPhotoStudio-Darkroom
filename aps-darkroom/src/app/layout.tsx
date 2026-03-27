import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "APS Darkroom",
  description: "Auto Photo Studio — The Darkroom",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="antialiased">{children}</body>
    </html>
  );
}
