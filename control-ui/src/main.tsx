import ReactDOM from 'react-dom/client';
import App from './App';
import LocalCollectionView from './local/LocalCollectionView';
import './styles.css';
import './operator-layout.css';
ReactDOM.createRoot(document.getElementById('root')!).render(new URLSearchParams(location.search).get('local')==='1'?<LocalCollectionView/>:<App />);
