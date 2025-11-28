import React, { useState, useContext, useEffect } from "react";
import { getCurrentUser } from "src/services/auth";

export const AuthContext = React.createContext({
  user: null,
  setUser: () => {},
  loading: true,
});

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchCurrentUser = async () => {
      const token = localStorage.getItem('token');
      
      if (!token) {
        setLoading(false);
        return;
      }

      try {
        const response = await getCurrentUser();
        setUser({ ...response, token });
      } catch (error) {
        console.error("Failed to fetch current user:", error);
        setUser(null);
        localStorage.removeItem('token');
        localStorage.removeItem('userId');
      } finally {
        setLoading(false);
      }
    };

    fetchCurrentUser();
  }, []);

  return (
    <AuthContext.Provider value={{ user, setUser, loading }}>
      {!loading && children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => useContext(AuthContext);
