import { useState } from "react";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

export const ReactQueryClientProvider = ({ children }) => {
  const [queryClient] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: {
            // With SSR, we usually want to set some default staleTime
            // above 0 to avoid refetching immediately on the client
            staleTime: 60 * 1000,
            retry: false,
          },
          mutations: {
            // with offline-first mode we will get an error if user is offline
            // default mode is "always", that causes infinite loading if user lost connection
            networkMode: "offlineFirst",
          },
        },
      })
  );
  return (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
};
