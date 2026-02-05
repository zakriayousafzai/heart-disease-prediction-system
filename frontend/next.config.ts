/** @type {import('next').NextConfig} */
const nextConfig = {
  async redirects() {
    return [
      {
        source: "/",
        destination: "/predict",
        permanent: false,
      },
    ];
  },
};

module.exports = nextConfig;
