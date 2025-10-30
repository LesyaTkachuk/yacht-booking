import { ThemeProvider } from "@mui/material/styles";
import { theme } from "src/theme";
import CssBaseline from "@mui/material/CssBaseline";
import { Snackbar, PageWrapper } from "src/components";
import AppNavigator from "src/navigation/AppNavigator";
import { AuthProvider } from "src/context/authContext";

function App() {

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Snackbar />
      <AuthProvider>
        <PageWrapper>
          <AppNavigator />
        </PageWrapper>
      </AuthProvider>
    </ThemeProvider>
  );
}

export default App;
