import type { Metadata } from "next";
import { Oswald, JetBrains_Mono } from "next/font/google";
import "./globals.css";

// Oswald: condensed, scoreboard/jersey-numeral feel for headings — a deliberate
// choice tied to the subject (courtside signage, not a generic SaaS landing page).
const oswald = Oswald({
  variable: "--font-display",
  subsets: ["latin"],
  weight: ["500", "600", "700"],
});

// JetBrains Mono for the tool-call trail — it's a log/terminal readout, so it
// should read like one instead of matching the body copy font.
const jetbrainsMono = JetBrains_Mono({
  variable: "--font-mono",
  subsets: ["latin"],
  weight: ["400", "500"],
});

export const metadata: Metadata = {
  title: "hoop-scout",
  description: "Agentic basketball scouting analyst",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="en"
      className={`${oswald.variable} ${jetbrainsMono.variable} h-full antialiased`}
    >
      <body className="min-h-full flex flex-col bg-court-ivory text-ink">{children}</body>
    </html>
  );
}
